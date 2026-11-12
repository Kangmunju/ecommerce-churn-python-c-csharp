using Churn.Core.Models;


namespace Churn.Core.Modeling;


// 은닉층 (32, 16) MLP 로 이탈 확률을 예측함
// 파이썬 MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, n_iter_no_change=20) 와 동일한 설정
// 같은 seed 면 같은 결과가 나옴 (난수는 sklearn 과 달라서 값이 똑같지는 않음)
// sealed: 더 이상 상속하지 못하게 막음
// : IClassifier 는 Fit 과 PredictProbability 를 꼭 만들겠다는 약속이라 교차검증에 그대로 들어감
public sealed class MlpClassifier : IClassifier
{
    private readonly int[] _hidden;
    private readonly int _seed;
    private readonly double _alpha;
    private readonly double _learningRate;
    private readonly int _maxEpochs;
    private readonly int _patience;
    private readonly double _tolerance;
    private readonly double _validationFraction;
    private readonly int _batchSize;

    // 표준화는 로지스틱 회귀와 같음
    // 학습 고객으로만 평균과 표준편차를 구함
    private readonly StandardScaler _scaler = new();

    // ? 가 붙은 MlpNetwork? 는 비어 있을 수(null) 있다는 뜻, Fit 을 하기 전에는 null 임
    private MlpNetwork? _network;

    // 매개변수에 = 값 이 붙어 있으면 기본값이라 생략해도 됨 (new MlpClassifier(5) 처럼)
    // int[]? hidden = null: 배열 기본값은 여기 직접 못 적어서 null 이면 안에서 { 32, 16 } 으로 정함
    public MlpClassifier(
        int seed = 0,
        int[]? hidden = null,
        double alpha = 1e-3,
        double learningRate = 1e-3,
        int maxEpochs = 600,
        int patience = 20,
        double tolerance = 1e-4,
        double validationFraction = 0.1,
        int batchSize = 200
    )
  {
      if(alpha < 0.0)
    {
        throw new ArgumentOutOfRangeException(nameof(alpha), "alpha 는 0 이상이어야 함");
    }
    if (validationFraction <= 0.0 || validationFraction >= 1.0)
    {
        throw new ArgumentOutOfRangeException(nameof(validationFraction), "검증 비율은 0 과 1 사이여야 함");
    }
    if (batchSize < 1)
    {
        throw new ArgumentOutOfRangeException(nameof(batchSize), "배치 크기는 1 이상이어야 함");
    }

    // (int[])hidden.Clone(): 배열을 복사함, 밖에서 원래 배열을 바꿔도 모델 설정은 그대로
    if (hidden == null)
    {
        _hidden = new[] { 32, 16 };
    }
    else
    {
        _hidden = (int[])hidden.Clone();
    }
    _seed = seed;
    _alpha = alpha;
    _learningRate = learningRate;
    _maxEpochs = maxEpochs;
    _patience = patience;
    _tolerance = tolerance;
    _validationFraction = validationFraction;
    _batchSize = batchSize;
  }

  // 실제로 학습한 에폭 수
  // 조기 종료되면 maxEpochs 보다 작음
  public int Epochs { get; private set; }

  // 가장 좋았던 검증 정확도
  // 이때의 가중치가 최종 모델이 됨
  public double BestValidationAccuracy { get; private set; }

  // 학습 순서
  // 1. 학습 고객으로 표준화하고 이 중 약 10% 는 검증용으로 따로 떼어 둠 (클래스 비율 유지)
  // 2. 가중치를 무작위로 시작함
  // 3. 학습용 고객을 섞어 200명씩 묶음(미니배치)으로 나눔
  // 4. 묶음마다 순전파 -> 역전파로 기울기를 구하고 평균을 내서 Adam 으로 가중치를 고침
  // 5. 한 바퀴(에폭)가 끝나면 검증 정확도를 잼 -> 20번 연속 좋아지지 않으면 멈춤
  // 6. 검증 정확도가 가장 좋았을 때의 가중치로 되돌림
  public void Fit(IReadOnlyList<CustomerFeatures> train)
  {
      if (train.Count == 0)
    {
        throw new ArgumentException("학습할 고객이 없음", nameof(train));
    }

    _scaler.Fit(train);

    int n = train.Count;
    var inputs = new double[n][];
    var labels = new double[n];
    for (int i = 0 ; i < n ; i++)
    {
        inputs[i] = _scaler.Transform(train[i]);
        if (train[i].Churn)
      {
          labels[i] = 1.0;
      }
    }

    var random = new Random(_seed);
    // var (a, b) = ...: 이름 붙은 튜플로 돌려받은 리스트 두 개를 한꺼번에 풀어서 받음
    var (fitIndexes, validationIndexes) = SplitValidation(labels, random);

    var sizes = new List<int> { FeatureNames.All.Count };
    sizes.AddRange(_hidden);
    sizes.Add(1);
    var network = new MlpNetwork(sizes, random);

    var adam = new AdamOptimizer(network.ParameterCount, _learningRate);
    var gradients = new double[network.ParameterCount];
    var activations = new double[sizes.Count][];

    // double.NegativeInfinity: 음의 무한대, 첫 에폭의 점수가 무엇이든 최고 기록이 되게 하는 시작값
    double bestScore = double.NegativeInfinity;
    double[] bestParameters = (double[])network.Parameters.Clone();
    int noImprovement = 0;
    int epochs = 0;

    for (int epoch = 0 ; epoch < _maxEpochs ; epoch++)
    {
        Shuffle(fitIndexes, random);

        for (int start = 0 ; start < fitIndexes.Count ; start += _batchSize)
      {
          int end = Math.Min(start + _batchSize, fitIndexes.Count);
          int size = end - start;

          // Array.Clear: 배열을 전부 0 으로 비움, 묶음마다 기울기를 새로 모으기 위함
          Array.Clear(gradients);
          for (int p = start ; p < end ; p++)
        {
            int index = fitIndexes[p];
            network.Forward(inputs[index], activations);
            network.Backward(activations, labels[index], gradients);
        }

        // 묶음 안 고객들의 기울기 합에 규제를 더하고 묶음 크기로 나눔
        network.AddWeightDecay(gradients, _alpha);
        for (int k = 0 ; k < gradients.Length ; k++)
        {
            gradients[k] /= size;
        }
        adam.Update(network.Parameters, gradients);
      }
      epochs++;

      double score = Accuracy(network, inputs, labels, validationIndexes);
      // 최고 기록보다 tolerance(1e-4) 이상 올라야 좋아진 것으로 보고, 아니면 정체 횟수를 셈
      // Array.Copy(원본, 대상, 길이): 배열 값을 복사함, 가장 좋았던 가중치를 따로 저장해 둠
      if (score < bestScore + _tolerance)
      {
          noImprovement++;
      }
      else
      {
          noImprovement = 0;
      }
      if (score > bestScore)
      {
          bestScore = score;
          Array.Copy(network.Parameters, bestParameters, bestParameters.Length);
      }
      if (noImprovement > _patience)
      {
          break;
      }
    }

    Array.Copy(bestParameters, network.Parameters, bestParameters.Length);
    Epochs = epochs;
    BestValidationAccuracy = bestScore;
    _network = network;
  } 

  public double PredictProbability(CustomerFeatures customer)
  {
      if (_network == null)
    {
        throw new InvalidOperationException("Fit 을 먼저 호출해야 함");
    }
    return _network.Predict(_scaler.Transform(customer));
  }

  // 이탈, 유지를 따로 섞어서 각각 validationFraction 만큼 검증용으로 뗌
  // 어느 쪽이든 최소 1명은 검증용에 들어가고 학습용에도 1명 이상 남도록 함
  private (List<int> Fit, List<int> Validation) SplitValidation(double[] labels, Random random)
  {
      var positives = new List<int>();
      var negatives = new List<int>();
      for (int i = 0 ; i < labels.Length ; i++)
    {
        if (labels[i] > 0.5)
      {
          positives.Add(i);
      }
      else
      {
          negatives.Add(i);
      }
    }
    if (positives.Count <2 || negatives.Count < 2)
    {
        throw new ArgumentException("이탈, 유지 고객이 각각 2명 이상 있어야 함");
    }

    Shuffle(positives, random);
    Shuffle(negatives, random);

    var fit = new List<int>();
    var validation = new List<int>();

    // new[] { positives, negatives }: 리스트 두 개를 배열에 담아 같은 코드로 한 번씩 처리함
    foreach (var group in new[] { positives, negatives })
    {
        int take = (int)Math.Round(group.Count * _validationFraction);
        take = Math.Max(1, Math.Min(take, group.Count - 1));
        for (int p = 0 ; p < group.Count ; p++)
      {
          if (p < take)
        {
            validation.Add(group[p]);
        }
        else
        {
            fit.Add(group[p]);
        }
      }
    }
    return (fit, validation);
  }

  // 확률이 0.5 보다 크면 이탈이라고 보고 맞힌 비율을 구함
  private static double Accuracy(MlpNetwork network, double[][] inputs, double[] labels, List<int> indexes)
  {
      int correct = 0;
      foreach (int index in indexes)
    {
        bool predicted = network.Predict(inputs[index]) > 0.5;
        bool actual = labels[index] > 0.5;
        if (predicted == actual)
      {
          correct++;
      }
    }
    // (double): 정수끼리 나누면 소수점이 잘려서 앞에 붙임 (3 / 4 = 0, 3.0 / 4 = 0.75)
    return (double)correct / indexes.Count;
  }

  private static void Shuffle(List<int> items, Random random)
  {
      for (int i = items.Count - 1 ; i > 0 ; i--)
    {
        int j = random.Next(i + 1);
        (items[i], items[j]) = (items[j], items[i]);
    }
  }
}

// ==================================================================================================
// [설계 메모]
// ==================================================================================================
// 1. 조기종료는 검증 정확도로 판단함 (sklearn early_stopping=True 와 같은 방식)
//    검증용으로 뗀 10% 고객은 학습에 쓰지 않고, 에폭마다 정확도만 잼
//    최고 기록보다 tol(1e-4) 이상 오르지 못한 에폭이 patience(20) 번을 넘게 이어지면 멈추고
//    가장 좋았던 때의 가중치로 되돌림 (마지막 에폭의 가중치가 아님)
//    정확도는 확률이 0.5 를 넘는지만 보기 때문에 쉬운 문제에서는 확률이 0.5 근처인 채로 멈출 수 있음
//    그래서 쉬운 문제 시험은 확률 크기가 아니라 AUC 와 0.5 기준 평균으로 확인함
//
// 2. 기울기 = (고객별 기울기의 합 + alpha * 가중치) / 묶음 크기
//    규제 항은 합에 더한 뒤 같이 나눔 (sklearn 과 같은 방식), 편향에는 규제를 걸지 않음
//
// 3. sklearn 과 값이 똑같지 않은 이유
//    난수 생성기가 달라 시작 가중치, 섞는 순서, 검증용으로 떼는 고객이 다름
//    그래서 시드 하나의 AUC 가 아니라 시드 10개 평균을 파이썬 범위(0.7836 +- 0.0035)와 비교함
//
// 4. 표준화가 없으면 MLP 가 크게 나빠짐 (파이썬 a6 에서 AUC 0.7886 -> 0.7189)
//    피처마다 크기가 수백 배 다르면 큰 피처가 학습을 끌고 가기 때문이라 Fit 안에서 꼭 표준화함
//    교차검증에서 조각마다 새 모델을 만드니 표준화도 학습 조각으로만 구해져 누수가 없음
