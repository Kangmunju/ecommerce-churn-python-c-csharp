namespace Churn.Core.Modeling;

// 완전연결 신경망 (입력 -> 은닉층들 -> 출력 1개)
// 은닉층 ReLU
// 출력 시그모이드
// 로그손실
// 모든 가중치와 편향을 Parameters 배열 하나에 펴서 담음
// Adam 과 기울기 확인이 단순해지도록 함
public sealed class MlpNetwork
{
    // 층마다 뉴런 수
    private readonly int[] _sizes;

    // l번째 연결(l층 -> l+1층)의 가중치와 편향이 Parameters 의 어디서 시작하는 지 구함
    // 가중치는 입력 i, 출력 j 일 때 weightStart[l] + i * 출력수 + j 의 식을 세움
    private readonly int[] _weightStart;
    private readonly int[] _biasStart;

    public double[] Parameters { get; }

    public int ParameterCount => Parameters.Length;

    // 연결의 개수
    // 층 수 - 1
    private int Links => _sizes.Length - 1;

    public MlpNetwork(IReadOnlyList<int> sizes, Random random)
    {
        if (sizes.Count < 2)
        {
            throw new ArgumentException("입력층과 출력층이 필요함", nameof(sizes));
        }
        if (sizes[sizes.Count - 1] != 1)
        {
            throw new ArgumentException("출력은 1개여야 함", nameof(sizes));
        }

        _sizes = sizes.ToArray();
        _weightStart = new int[Links];
        _biasStart = new int[Links];

        int total = 0;
        for (int l = 0; l < Links; l++)
        {
            _weightStart[l] = total;
            total += _sizes[l] * _sizes[l + 1];
            _biasStart[l] = total;
            total += _sizes[l + 1];
        }
        Parameters = new double[total];

        // 처음 값: -bound ~ +bound 사이 무작위 값으로 설정함 (Glorot 초기화)
        // 층이 넓을 수록 bound 가 작아져서 층을 지나도 값의 크기가 비슷하게 유지되도록 함
        for (int l = 0; l < Links; l++)
        {
            double bound = Math.Sqrt(6.0 / (_sizes[l] + _sizes[l + 1]));
            for (int k = _weightStart[l]; k < _biasStart[l] + _sizes[l + 1]; k++)
            {
                Parameters[k] = (random.NextDouble() * 2.0 - 1.0) * bound;
            }
        }
    }

    // 한 명의 순전파
    // activations[l]: l번째 층의 출력 (0번은 입력 그대로), 호출하는 쪽이 층 수만큼 만들어서 넘김
    // 돌려주는 값: 이탈 확률
    public double Forward(double[] input, double[][] activations)
    {
        if (input.Length != _sizes[0])
        {
            throw new ArgumentException("입력 개수가 맞지 않음", nameof(input));
        }

        activations[0] = input;
        for (int l = 0; l < Links; l++)
        {
            int fanIn = _sizes[l];
            int fanOut = _sizes[l + 1];
            var previous = activations[l];
            var current = new double[fanOut];

            for (int j = 0; j < fanOut; j++)
            {
                // z = 편향 + (앞 층 출력* 가중치) 의 합
                double z = Parameters[_biasStart[l] + j];
                for (int i = 0; i < fanIn; i++)
                {
                    z += previous[i] * Parameters[_weightStart[l] + i * fanOut + j];
                }

                if (l == Links - 1)
                {
                    current[j] = Sigmoid(z);
                }
                else
                {
                    current[j] = Math.Max(0.0, z);
                }
            }
            activations[l + 1] = current;
        }
        return activations[Links][0];
    }

    // 고객 한 명의 이탈 확률
    public double Predict(double[] input)
    {
        var activations = new double[_sizes.Length][];
        return Forward(input, activations);
    }

    // 한 명의 역전파
    // Forward 로 채운 activations 와 정답(1 이탈, 0 유지)으로 기울기를 구해 gradients 에 더함
    // 미니배치의 여러 명을 같은 gradients 에 계속 더해 가기 위함임
    public void Backward(double[][] activations, double label, double[] gradients)
    {
        // 출력층의 delta = 예측 확률 - 정답
        // 시그모이드와 로그 손실을 같이 쓰면 미분이 이 한 줄로 정리됨
        var delta = new double[] { activations[Links][0] - label };

        for (int l = Links - 1; l >= 0; l--)
        {
            int fanIn = _sizes[l];
            int fanOut = _sizes[l + 1];
            var previous = activations[l];

            // 이 연결의 가중치 기울기 = 앞 층 출력 * delta, 편향 기울기 = delta
            for (int j = 0; j < fanOut; j++)
            {
                gradients[_biasStart[l] + j] += delta[j];
                for (int i = 0; i < fanIn; i++)
                {
                    gradients[_weightStart[l] + i * fanOut + j] += previous[i] * delta[j];
                }
            }

            // 앞 층의 delta = (가중치를 따라 거슬러 올라온 delta 의 합) * ReLU 미분
            // ReLU 미분: 출력이 0 보다 크면 1, 아니면 0 (입력층은 필요 없음)
            if (l > 0)
            {
                var previousDelta = new double[fanIn];
                for (int i = 0; i < fanIn; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < fanOut; j++)
                    {
                        sum += Parameters[_weightStart[l] + i * fanOut + j] * delta[j];
                    }
                    if (previous[i] > 0.0)
                    {
                        previousDelta[i] = sum;
                    }
                }
                delta = previousDelta;
            }
        }
    }

    // 규제(alpha)의 기울기를 더함
    // 가중치마다 alpha * 가중치 (편향은 제외함)
    // 가중치가 커질수록 손실에 벌점을 주어 값이 커지는 것을 막음
    public void AddWeightDecay(double[] gradients, double alpha)
    {
        for (int l = 0; l < Links; l++)
        {
            for (int k = _weightStart[l]; k < _biasStart[l]; k++)
            {
                gradients[k] += alpha * Parameters[k];
            }
        }
    }

    // 가중치(편향 제외)의 제곱합
    // 규제가 손실에 더하는 값은 0.5 * alpha * 이 값으로 계산함
    public double SumOfSquaredWeights()
    {
        double sum = 0.0;
        for (int l = 0; l < Links; l++)
        {
            for (int k = _weightStart[l]; k < _biasStart[l]; k++)
            {
                sum += Parameters[k] * Parameters[k];
            }
        }
        return sum;
    }

    private static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }
}
