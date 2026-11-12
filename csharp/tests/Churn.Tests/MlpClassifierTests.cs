using Churn.Core.Evaluation;
using Churn.Core.Models;
using Churn.Core.Modeling;
using Churn.Core.Validation;


namespace Churn.Tests;


// MlpClassifier 의 시험
// 쉬운 가짜 문제로 학습, 조기 종료, 같은 seed 재현율을 확인
// 실제 데이터는 파이썬 MLP(시드 10개 평균 0.7836 +- 0.0035) 범위와 비교함
public class MlpClassifierTests
{

    // 피처 0번 하나만 정답과 관련 있는 가짜 고객 count 명
    // 이탈이면 +2 근처
    // 유지면 -2 근처
    // 나머지 피처 19개는 0
    private static List<CustomerFeatures> EasyCustomers(int count, int seed)
  {
      var random = new Random(seed);
      var customers = new List<CustomerFeatures>();
      for (int i = 0 ; i < count ; i++)
    {
        bool churn = i % 2 == 0;
        var values = new double[FeatureNames.All.Count];
        double center = -2.0;
        if (churn)
      {
          center = 2.0;
      }
      values[0] = center + (random.NextDouble() - 0.5);
      customers.Add(new CustomerFeatures(i + 1, values, churn, false));
    }
    return customers;
  }

  // 쉬운 문제는 순서를 완벽하게 맞혀야 함 (AUC 가 거의 1)
  // 이탈 고객의 평균 확률은 0.5 보다 크고 유지 고객은 0.5 보다 작음
  // 조기종료가 검증 정확도로 판단하기 때문에 확률이 0.9 이상까지는 커지지는 않음
  // 정확도가 1 이 되면 확률이 아직 0.5 근처여도 20번 뒤에 멈춤
  [Fact]
  public void Fit_learns_an_easy_problem()
  {
      var train = EasyCustomers(300, 1);
      var test = EasyCustomers(100, 2);
      var model = new MlpClassifier(0);

      model.Fit(train);

      var probabilities = new List<double>();
      var churn = new List<bool>();
      foreach (var c in test)
    {
        double p = model.PredictProbability(c);
        Assert.InRange(p, 0.0, 1.0);
        probabilities.Add(p);
        churn.Add(c.Churn);
    }
    Assert.True(MetricsCalculator.Evaluate(churn, probabilities).Auc > 0.99);

    double churnMean = 0.0;
    double stayMean = 0.0;
    for (int i = 0 ; i < test.Count ; i++)
    {
        if (churn[i])
      {
          churnMean += probabilities[i] / 50.0;
      }
      else
      {
          stayMean += probabilities[i] / 50.0;
      }
    }
    Assert.True(churnMean > 0.5);
    Assert.True(stayMean < 0.5);
  }

  // 검증 정확도가 20번 연속 좋아지지 않으면 600번을 다 채우지 않고 멈춤
  [Fact]
  public void Fit_stops_early_when_validation_stops_improving()
  {
      var model = new MlpClassifier(0);

      model.Fit(EasyCustomers(300, 1));

      Assert.InRange(model.Epochs, 1, 599);
      Assert.True(model.BestValidationAccuracy >= 0.9);
  }

  // 최대 에폭을 3으로 제한하면 3번만 돌고 끝남
  [Fact]
  public void Fit_respects_the_max_epochs()
  {
      // maxEpochs: 3 처럼 이름을 붙여 넘기면 기본값이 있는 매개변수 중 이것만 골라 값을 줄 수 있음 (콜론 필수)
      var model = new MlpClassifier(0, maxEpochs: 3);

      model.Fit(EasyCustomers(300, 1));

      Assert.Equal(3, model.Epochs);
  }

  // 같은 seed 면 같은 확률
  // 다른 seed 면 다른 확률
  [Fact]
  public void Fit_is_repeatable_for_the_same_seed()
  {
      
      // 300명(이탈 150, 유지 150), 1 은 가짜 데이터를 만드는 난수 시드
      // 300명 중 약 10% 인 30명이 검증용으로 빠지고 270명으로 학습함 (200명 묶음 1개 + 70명 묶음 1개)
      var train = EasyCustomers(300, 1);

      // 학습에 쓰지 않은 고객 한 명이 필요해서 시드를 3 으로 달리해 만듦 (학습 1, 시험 2, 여기 3)
      // 5명은 [1] 번 자리가 있기만 하면 되는 개수임
      // [1] 은 i 가 1 이라 유지 고객
      var probe = EasyCustomers(5, 3)[1];

      // (5), (5), (6) 은 모델의 난수 시드
      // (5), (5) 는 같은 값이기 때문에 결과가 동일해야 함
      // (6) 은 다른 값이기 때문에 결과가 달라야 함
      // 5, 6 숫자 자체에 의미가 없음에 주의
      var first = new MlpClassifier(5);
      var again = new MlpClassifier(5);
      var other = new MlpClassifier(6);

      first.Fit(train);
      again.Fit(train);
      other.Fit(train);

      Assert.Equal(first.PredictProbability(probe), again.PredictProbability(probe));
      Assert.NotEqual(first.PredictProbability(probe), other.PredictProbability(probe));
  }

  [Fact]
  public void PredictProbability_requires_fit()
  {
      var model = new MlpClassifier(0);

      Assert.Throws<InvalidOperationException>(() => model.PredictProbability(EasyCustomers(2, 1)[0]));
  }

  [Fact]
  public void Fit_rejects_empty_input_and_single_class_input()
  {
      var model = new MlpClassifier(0);
      var onlyChurn = new List<CustomerFeatures>();
      for (int i = 0 ; i < 10 ; i++)
    {
        onlyChurn.Add(new CustomerFeatures(i + 1, new double[FeatureNames.All.Count], true, false));
    }

    Assert.Throws<ArgumentException>(() => model.Fit(new List<CustomerFeatures>()));
    Assert.Throws<ArgumentException>(() => model.Fit(onlyChurn));
  }

  [Fact]
  public void Constructor_rejects_bad_settings()
  {
      Assert.Throws<ArgumentOutOfRangeException>(() => new MlpClassifier(0, alpha: -1.0));
      Assert.Throws<ArgumentOutOfRangeException>(() => new MlpClassifier(0, validationFraction: 0.0));
      Assert.Throws<ArgumentOutOfRangeException>(() => new MlpClassifier(0, validationFraction: 1.0));
      Assert.Throws<ArgumentOutOfRangeException>(() => new MlpClassifier(0, batchSize: 0));
  }

  // 실제 데이터로 시드 0 ~ 9 를 학습시켜 시험셋 AUC 의 평균을 구함
  // 파이썬은 시드 10개 평균 0.7836, 표준편차 0.0035
  // 난수가 sklearn 과 달라 시드마다 값은 다르지만 평균은 같은 범위 안에 있어야 함
  // 시드 10 ~ 39 로 30개를 더 돌려도 평균이 0.7858 근처로 확인됨
  [DataFact]
  public void Fit_gives_test_auc_close_to_python_on_real_data()
  {
      var churn = TestData.Test.Select(c => c.Churn).ToList();
      var aucs = new List<double>();

      for (int seed = 0 ; seed < 10 ; seed++)
    {
        var model = new MlpClassifier(seed);
        model.Fit(TestData.Train);
        var probabilities = TestData.Test.Select(c => model.PredictProbability(c)).ToList();
        aucs.Add(MetricsCalculator.Evaluate(churn, probabilities).Auc);
    }

    Assert.InRange(aucs.Average(), 0.7836 - 0.0035, 0.7836 + 0.0035);
    Assert.All(aucs, auc => Assert.InRange(auc, 0.75, 0.82));
  }

  // MLP 도 IClassifier 라서 8단계 교차 검증에 그대로 들어감
  // 파이썬 MLP 의 5-fold cv 평균은 0.8046
  // 여기서는 시드 하나만 쓰기 때문에 넓게 잡았음
  [DataFact]
  public void CrossValidator_runs_with_mlp_on_real_data()
  {
      var all = TestData.Train.Concat(TestData.Test).OrderBy(c => c.CustomerId).ToList();

      var result = CrossValidator.Run(all, () => new MlpClassifier(0), 5, 42);

      Assert.InRange(result.MeanAuc, 0.79, 0.82);
      Assert.Equal(4944, result.OofProbabilities.Count);
  }
}