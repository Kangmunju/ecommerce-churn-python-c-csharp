using System.Globalization;
using Churn.Core.Models;
using Churn.Core.Modeling;


namespace Churn.Tests;


// LogisticRegression 의 시험
// 기대 계수는 tools/expected_values.py 의 logistic() 출력
// sklearn tol=1e-12, 학습 3,460명
public class LogisticRegressionTests
{

    // 표준화된 피처에 대한 계수
    // Feature 번호 순서
    private static readonly double[] ExpectedCoefficients =
  {
      +0.4506646029, -0.0889310089, +0.0520370798, -0.1251122816, -0.0736773152,
      +0.1633841582, -0.1691433271, +0.0140261778, -0.3081052258, +0.2612149893,
      +0.0732026197, +0.1435904920, +0.0613287830, -1.4657474484, +0.1853803167,
      +0.3032913514, -0.4637288594, +0.1934765540, -0.1370722240, -0.0464534615,
  };

  private const double ExpectedIntercept = -0.3340369420;

  // 시험용 고객: 첫 번째, 두 번째 피처만 값을 주고 나머지는 0을 줌
  private static CustomerFeatures Make(int id, double first, double second, bool churn)
  {
      var values = new double[FeatureNames.All.Count];
      values[0] = first;
      values[1] = second;
      return new CustomerFeatures(id, values, churn, false);
  }

  // 정답이 피처 값과 완전히 갈라지지 않도록 섞어 둔 고객 10명
  private static List<CustomerFeatures> Sample()
  {
      double[] first = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
      double[] second = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
      bool[] churn = { false, true, false, false, true, false, true, true, false, true };

      var customers = new List<CustomerFeatures>();
      for (int i = 0 ; i < first.Length ; i++)
    {
        customers.Add(Make(i + 1, first[i], second[i], churn[i]));
    }
    return customers;
  }

  // 최적점에서는 L 의 기울기가 모두 0 이어야 함
  // 모델 안쪽 계산을 빌리지 않고 예측 확률과 계수만으로 기울기를 다시 계산해서 확인함
  // 계수 k: 합계((p - y) * z_k) + (1 / C) * w_k
  // 절편: 합계(p - y)
  [Fact]
  public void Fit_reaches_zero_gradient()
  {
      var customers = Sample();
      var model = new LogisticRegression(1.0);

      model.Fit(customers);

      int features = FeatureNames.All.Count;
      var grad = new double[features + 1];
      foreach (var c in customers)
    {
        double y = 0.0;
        if (c.Churn)
      {
          y = 1.0;
      }
      double err = model.PredictProbability(c) - y;
      var z = model.Scaler.Transform(c);

      for (int k = 0 ; k < features ; k++)
      {
          grad[k] += err * z[k];
      }
      grad[features] += err;
    }

    for (int k = 0 ; k < features ; k++)
    {
        grad[k] += model.Coefficients[k];
    }

    foreach (var g in grad)
    {
        Assert.True(Math.Abs(g) < 1e-8, $"기울기 {g:E2}");
    }
  }

  // C 가 작을수록 규제가 세서 계수가 작아져야 함
  [Fact]
  public void Fit_shrinks_coefficients_with_stronger_regularization()
  {
      var weak = new LogisticRegression(1.0);
      var strong = new LogisticRegression(0.01);

      weak.Fit(Sample());
      strong.Fit(Sample());

      Assert.True(SumSquares(strong.Coefficients) < SumSquares(weak.Coefficients));
  }

  [Fact]
  public void PredictProbability_stays_between_zero_and_one()
  {
      var model = new LogisticRegression();
      model.Fit(Sample());

      foreach (var c in Sample())
    {
        double p = model.PredictProbability(c);
        Assert.InRange(p, 0.0, 1.0);
    }

    // 학습 범위를 한참 벗어난 값에서도 넘치지 않고 0 과 1 사이여야 함
    double far = model.PredictProbability(Make(99, 1e6, -1e6, false));
    Assert.InRange(far, 0.0, 1.0);
  }

  [Fact]
  public void Fit_rejects_empty_list()
  {
      Assert.Throws<ArgumentException>(() => new LogisticRegression().Fit(new List<CustomerFeatures>()));
  }

  // Fit 으로 계수를 구하기 전에는 예측할 수 없음
  [Fact]
  public void PredictProbability_requires_fit()
  {
      Assert.Throws<InvalidOperationException>(() => new LogisticRegression().PredictProbability(Make(1, 1.0, 1.0, false)));
  }

  // C 가 0 이하이면 1 / C 로 규제 세기를 구할 수 없으므로 거부함
  [Fact]
  public void Constructor_rejects_non_positive_c()
  {
      Assert.Throws<ArgumentOutOfRangeException>(() => new LogisticRegression(0.0));
  }

  // 끝까지 수렴한 sklearn 값과 비교함
  // 뉴턴법이 10회 안팎에 끝나는지도 여기서 확인함
  [DataFact]
  public void Fit_matches_converged_sklearn_coefficients()
  {
      var model = new LogisticRegression(1.0);

      model.Fit(TestData.Train);

      Assert.True(model.Iterations < 20, $"반복 {model.Iterations}회");
      Assert.Equal(ExpectedIntercept, model.Intercept, 5);
      for (int k = 0 ; k < ExpectedCoefficients.Length ; k++)
    {
        Assert.True(
            Math.Abs(model.Coefficients[k] - ExpectedCoefficients[k]) < 1e-5,
            $"{FeatureNames.All[k]}: {model.Coefficients[k]:F10} vs {ExpectedCoefficients[k]:F10}"
        );
    }
  }

  // reference_test_pred.csv 의 p_logit 은 sklearn 기본 허용 오차에서 멈춘 값임
  // 이 값을 시험 고객별로 비교함
  // 정답 CSV 가 최적점에서 조금 떨어져 있으므로 확률 차이 1e-2 까지 허용함
  [DataFact]
  public void PredictProbability_matches_reference_test_pred()
  {
      var model = new LogisticRegression(1.0);
      model.Fit(TestData.Train);

      // 고객 번호로 정답 확률을 바로 찾도록 Dictionary 에 담음
      // 파일 열 순서: customer_id, y, p_logit (f[0], f[1], f[2])
      var expected = new Dictionary<int, double>();
      foreach (var line in File.ReadLines(TestData.ReferenceTestPred).Skip(1))
    {
        var f = line.Split(',');
        expected.Add(int.Parse(f[0]), double.Parse(f[2], CultureInfo.InvariantCulture));
    }
    Assert.Equal(expected.Count, TestData.Test.Count);

    // 시험 고객마다 C# 확률과 정답 확률의 차이를 구하고 가장 큰 값을 기록함
    double maxDiff = 0.0;
    foreach (var c in TestData.Test)
    {
        double diff = Math.Abs(model.PredictProbability(c) - expected[c.CustomerId]);
        if (diff > maxDiff)
      {
          maxDiff = diff;
      }
    }
    Assert.True(maxDiff < 1e-2, $"최대 확률 차이 {maxDiff:E2}");
  }

  // 계수들의 제곱합, 계수가 전체적으로 얼마나 큰지 재는 값
  private static double SumSquares(IReadOnlyList<double> values)
  {
    double sum = 0.0;
    foreach (var v in values)
    {
        sum += v * v;
    }
    return sum;
  }
}