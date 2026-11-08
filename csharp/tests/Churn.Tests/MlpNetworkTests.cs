using Churn.Core.Modeling;

namespace Churn.Tests;

// MlpNetwork 의 시험
// 아주 작은 신경망(입력 2, 은닉 2, 출력 1)을 손으로 계산한 값과 맞춰 봄
// 역전파가 맞는지는 수치 미분(기울기를 직접 재 본 값)과 비교해서 확인함
public class MlpNetworkTests
{
    // 입력 2, 은닉 2, 출력 1 신경망에 손으로 정한 가중치를 넣음
    // Parameters 순서: 연결0 가중치 4개 (w00, w01, w10, w11), 연결0 편향 2개, 연결1 가중치 2개, 연결1 편향 1개
    // 가중치 w(i, j) 는 입력 i 에서 출력 j 로 가는 값
    private static MlpNetwork TinyNetwork()
    {
        var network = new MlpNetwork(new[] { 2, 2, 1 }, new Random(0));
        double[] values = { 1.0, -1.0, 2.0, 0.5, 0.5, -3.0, 0.2, 0.7, -1.0 };
        for (int k = 0; k < values.Length; k++)
        {
            network.Parameters[k] = values[k];
        }
        return network;
    }

    private static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    // 총 20 -> 32 -> 16 -> 1 이면 (20 x 32 + 32) + (32 x 16 + 16) + (16 x 1 + 1) = 1,217
    [Fact]
    public void ParameterCount_counts_weights_and_biases()
    {
        var network = new MlpNetwork(new[] { 20, 32, 16, 1 }, new Random(0));

        Assert.Equal(1217, network.ParameterCount);
        Assert.Equal(1217, network.Parameters.Length);
    }

    // 입력 (1, 2)
    // 은닉 0번: 0.5 + 1 x 1 + 2 x 2 = 5.5 -> ReLU 5.5
    // 은닉 1번: -3 + 1 x (-1) + 2 x 0.5 = -3 -> ReLU 0 (음수는 0 으로 잘림)
    // 출력: -1 + 0.2 x 5.5 + 0.7 x 0 = 0.1 -> 시그모이드 0.1
    [Fact]
    public void Forward_matches_hand_calculation()
    {
        var network = TinyNetwork();
        var activations = new double[3][];

        double p = network.Forward(new[] { 1.0, 2.0 }, activations);

        Assert.Equal(5.5, activations[1][0], 12);
        Assert.Equal(0.0, activations[1][1], 12);
        Assert.Equal(Sigmoid(0.1), p, 12);
        Assert.Equal(p, activations[2][0], 12);
    }

    [Fact]
    public void Predict_gives_the_same_probability_as_forward()
    {
        var network = TinyNetwork();

        Assert.Equal(Sigmoid(0.1), network.Predict(new[] { 1.0, 2.0 }), 12);
    }

    // 정답이 1 (이탈) 일 때 위 신경망의 기울기를 손으로 구함
    // 출력 delta = p - 1 = d
    // 출력 편향 기울기 = d, 출력 가중치 기울기 = 은닉 출력 x d = (5.5 d, 0)
    // 은닉 0번 delta = 0.2 x d (ReLU 미분 1), 은닉 1번 delta = 0 (출력이 0 이라 ReLU 미분 0)
    // 은닉 가중치 기울기 = 입력 x 은닉 delta, 은닉 편향 기울기 = 은닉 delta
    [Fact]
    public void Backward_matches_hand_calculation()
    {
        var network = TinyNetwork();
        var activations = new double[3][];
        double p = network.Forward(new[] { 1.0, 2.0 }, activations);
        var gradients = new double[network.ParameterCount];

        network.Backward(activations, 1.0, gradients);

        double d = p - 1.0;
        double h = 0.2 * d;
        double[] expected = { 1.0 * h, 0.0, 2.0 * h, 0.0, h, 0.0, 5.5 * d, 0.0, d };
        for (int k = 0; k < expected.Length; k++)
        {
            Assert.Equal(expected[k], gradients[k], 12);
        }
    }

    // Backward 는 덮어쓰지 않고 더함
    // 같은 고객을 두 번 넣으면 기울기가 2배가 됨
    [Fact]
    public void Backward_adds_to_existing_gradients()
    {
        var network = TinyNetwork();
        var activations = new double[3][];
        network.Forward(new[] { 1.0, 2.0 }, activations);
        var once = new double[network.ParameterCount];
        var twice = new double[network.ParameterCount];

        network.Backward(activations, 1.0, once);
        network.Backward(activations, 1.0, twice);
        network.Backward(activations, 1.0, twice);

        for (int k = 0; k < once.Length; k++)
        {
            Assert.Equal(2.0 * once[k], twice[k], 12);
        }
    }

    // 규제는 가중치에만 alpha * 가중치를 더하고 편향은 건드리지 않음
    [Fact]
    public void AddWeightDecay_changes_weights_but_not_biases()
    {
        var network = TinyNetwork();
        var gradients = new double[network.ParameterCount];

        network.AddWeightDecay(gradients, 0.1);

        // 가중치 자리: 0 ~ 3, 6 ~ 7 / 편향 자리: 4 ~ 5, 8
        Assert.Equal(0.1 * 1.0, gradients[0], 12);
        Assert.Equal(0.1 * 0.5, gradients[3], 12);
        Assert.Equal(0.1 * 0.7, gradients[7], 12);
        Assert.Equal(0.0, gradients[4]);
        Assert.Equal(0.0, gradients[5]);
        Assert.Equal(0.0, gradients[8]);
    }

    // 가중치 제곱합 = 1 + 1 + 4 + 0.25 + 0.04 + 0.49 = 6.78 (편향 0.5, -3, -1 은 제외)
    [Fact]
    public void SumOfSquaredWeights_excludes_biases()
    {
        Assert.Equal(6.78, TinyNetwork().SumOfSquaredWeights(), 12);
    }

    // 수치 미분 확인
    // 가중치 하나를 아주 조금(+-1e-6) 바꿔서 손실이 변하는 정도를 직접 잼
    // 역전파가 구한 기울기와 비교함
    // 역전파 식에 오류가 있어도 신경망은 돌아가고 AUC 도 그럴듯하게 나오기 때문에 꼭 하고 넘어가야 함
    // 손실 = 고객별 로그 손실의 합 + 0.5 x alpha x 가중치 제곱합 (기울기도 고객별 합에 alpha x 가중치를 더한 것)
    [Fact]
    public void Backward_matches_numerical_gradients()
    {
        var random = new Random(7);
        var network = new MlpNetwork(new[] { 5, 4, 3, 1 }, random);
        const double alpha = 0.3;

        var inputs = new List<double[]>();
        var labels = new List<double>();
        for (int n = 0; n < 6; n++)
        {
            var x = new double[5];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = random.NextDouble() * 4.0 - 2.0;
            }
            inputs.Add(x);
            labels.Add(n % 2);
        }

        // 역전파로 구한 기울기
        var analytic = new double[network.ParameterCount];
        var activations = new double[4][];
        for (int n = 0; n < inputs.Count; n++)
        {
            network.Forward(inputs[n], activations);
            network.Backward(activations, labels[n], analytic);
        }
        network.AddWeightDecay(analytic, alpha);

        // 가중치를 하나씩 조금 올리고 내려서 직접 잰 기울기
        const double step = 1e-6;
        double worst = 0.0;
        for (int k = 0; k < network.ParameterCount; k++)
        {
            double original = network.Parameters[k];

            network.Parameters[k] = original + step;
            double up = Loss(network, inputs, labels, alpha);
            network.Parameters[k] = original - step;
            double down = Loss(network, inputs, labels, alpha);
            network.Parameters[k] = original;

            double numeric = (up - down) / (2.0 * step);
            worst = Math.Max(worst, Math.Abs(numeric - analytic[k]));
        }

        Assert.True(worst < 1e-6, $"가장 큰 차이 {worst}");
    }

    private static double Loss(MlpNetwork network, List<double[]> inputs, List<double> labels, double alpha)
    {
        double sum = 0.0;
        for (int n = 0; n < inputs.Count; n++)
        {
            double p = network.Predict(inputs[n]);
            sum += -(labels[n] * Math.Log(p) + (1.0 - labels[n]) * Math.Log(1.0 - p));
        }
        return sum + 0.5 * alpha * network.SumOfSquaredWeights();
    }

    // 처음 값은 Glorot 범위 안
    // 첫 연결(20 -> 32)은 +-sqrt(6 / 52) = +-0.34
    // 같은 seed 면 같은 값, 다른 seed 면 다른 값이 나옴
    [Fact]
    public void Constructor_initializes_within_glorot_bound_and_is_repeatable()
    {
        var first = new MlpNetwork(new[] { 20, 32, 16, 1 }, new Random(1));
        var again = new MlpNetwork(new[] { 20, 32, 16, 1 }, new Random(1));
        var other = new MlpNetwork(new[] { 20, 32, 16, 1 }, new Random(2));

        // 첫 연결의 가중치 640개와 편향 32개
        double bound = Math.Sqrt(6.0 / 52.0);
        double largest = 0.0;
        for (int k = 0; k < 672; k++)
        {
            largest = Math.Max(largest, Math.Abs(first.Parameters[k]));
        }
        Assert.True(largest <= bound);
        Assert.True(largest > 0.9 * bound);

        Assert.Equal(first.Parameters, again.Parameters);
        Assert.NotEqual(first.Parameters, other.Parameters);
    }

    [Fact]
    public void Constructor_rejects_bad_layer_sizes()
    {
        Assert.Throws<ArgumentException>(() => new MlpNetwork(new[] { 3 }, new Random(0)));
        Assert.Throws<ArgumentException>(() => new MlpNetwork(new[] { 3, 4, 2 }, new Random(0)));
    }

    [Fact]
    public void Forward_rejects_wrong_input_length()
    {
        var network = TinyNetwork();

        Assert.Throws<ArgumentException>(() => network.Predict(new[] { 1.0, 2.0, 3.0 }));
    }
}
