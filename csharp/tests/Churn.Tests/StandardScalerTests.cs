using System.Globalization;
using Churn.Core.Models;
using Churn.Core.Modeling;


namespace Churn.Tests;


// StandardScaler 의 시험
// 작은 데이터는 손으로 계산한 값, 실제 데이터는 reference_coefficients.csv 의 mean, scale 과 비교함
public class StandardScalerTests
{

    // 시험용 고객: 첫 번째 피처와 두 번째 피처만 값을 주고 나머지 18개는 0으로 줌
    private static CustomerFeatures Make(int id, double first, double second)
  {
      var values = new double[FeatureNames.All.Count];
      values[0] = first;
      values[1] = second;
      return new CustomerFeatures(id, values, false, false);
  }

  private static List<CustomerFeatures> Sample()
  {
      return new List<CustomerFeatures>
      {
          Make(1, 2.0, 7.0),
          Make(2, 4.0, 7.0),
          Make(3, 6.0, 7.0),
          Make(4, 8.0, 7.0),
      };
  }

  // 2, 4, 6, 8 의 평균은 5
  // 편차 -3 -1 1 3 의 제곱합 20을 n(4)으로 나누면 5
  // 표준편차는 sqrt(5) 으로 구함
  [Fact]
  public void Fit_computes_mean_and_population_std()
  {
      var scaler = new StandardScaler();

      scaler.Fit(Sample());

      Assert.Equal(5.0, scaler.Mean[0], 12);
      Assert.Equal(Math.Sqrt(5.0), scaler.Scale[0], 12);
  }

  // 모든 고객이 같은 값이면 표준편차가 0 이라 나눌 수 없으므로 1로 둠
  [Fact]
  public void Fit_uses_one_for_constant_feature()
  {
      var scaler = new StandardScaler();

      scaler.Fit(Sample());

      Assert.Equal(7.0, scaler.Mean[1], 12);
      Assert.Equal(1.0, scaler.Scale[1]);
      Assert.Equal(1.0, scaler.Scale[5]);
  }

  // Sample() 로 학습한 평균 5, 표준편차 sqrt(5) 를 새 고객에게 적용함
  // 첫 번째 피처 4 -> (4 - 5) / sqrt(5)
  // 두 번째 피처 10 -> 평균 7, 표준편차 1(상수 피처) 이므로 (10 - 7) / 1 = 3
  [Fact]
  public void Transform_standardizes_with_fitted_statistics()
  {
      var scaler = new StandardScaler();
      scaler.Fit(Sample());

      var z = scaler.Transform(Make(9, 4.0, 10.0));

      Assert.Equal((4.0 - 5.0) / Math.Sqrt(5.0), z[0], 12);
      Assert.Equal(3.0, z[1], 12);
      Assert.Equal(20, z.Length);
  }

  // 고객이 한 명도 없으면 평균을 구할 수 없으므로 오류를 던짐
  [Fact]
  public void Fit_rejects_empty_list()
  {
      Assert.Throws<ArgumentException>(() => new StandardScaler().Fit(new List<CustomerFeatures>()));
  }

  // Fit 으로 평균을 구하기 전에는 표준화할 수 없음
  [Fact]
  public void Transform_requires_fit()
  {
      Assert.Throws<InvalidOperationException>(() => new StandardScaler().Transform(Make(1, 1.0, 1.0)));
  }

  // reference_coefficients.csv 의 feature, mean, scale, coef 중 mean 과 scale 을 비교함
  // 파이썬 StandardScaler 가 학습 고객 3,460명으로 구한 값과 같아야 함
  // 오차는 값이 클 때 상대 오차로 보도록 max(1, |기대값|) 로 나눔
  [DataFact]
  public void Fit_matches_reference_mean_and_scale()
  {
      // 학습 고객 수가 파이썬과 같은지 먼저 확인함
      Assert.Equal(3460, TestData.Train.Count);
      var scaler = new StandardScaler();

      scaler.Fit(TestData.Train);

      var lines = File.ReadLines(TestData.ReferenceCoefficients).Skip(1).ToList();
      Assert.Equal(FeatureNames.All.Count, lines.Count);

      for (int k = 0 ; k < lines.Count ; k++)
    {
        var f = lines[k].Split(',');
        double mean = double.Parse(f[1], CultureInfo.InvariantCulture);
        double scale = double.Parse(f[2], CultureInfo.InvariantCulture);

        // 파일의 줄 순서와 FeatureNames.All 의 순서가 같은지도 함께 확인함
        Assert.Equal(FeatureNames.All[k], f[0]);
        Assert.True(Math.Abs(scaler.Mean[k] - mean) / Math.Max(1.0, Math.Abs(mean)) < 1e-9, f[0]);
        Assert.True(Math.Abs(scaler.Scale[k] - scale) / Math.Max(1.0, Math.Abs(scale)) < 1e-9, f[0]);
    }
  }
}