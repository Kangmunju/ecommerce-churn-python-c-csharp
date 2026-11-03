using Churn.Core.Models;


namespace Churn.Core.Modeling;


// 피처마다 평균과 표준편차로 값을 표준화함
// z = (x - 평균) / 표준편차
// 평균과 표준편차는 학습 고객에서만 구하고 시험 고객에도 그대로 사용함
// 시험 고객의 통계를 그대로 사용하면 데이터 누수가 발생할 위험 있음
// 표준편차는 n으로 나눔
// sklearn StandardScaler 방식으로 피처를 만들 때 쓴 n-1 과 다른 방식을 선택함
public sealed class StandardScaler
{
    private double[] _mean = Array.Empty<double>();
    private double[] _scale = Array.Empty<double>();

    // Fit 을 마쳤으면 true 를 반환함
    public bool IsFitted => _mean.Length > 0;

    public IReadOnlyList<double> Mean => _mean;

    // 표준편차
    // 모든 고객이 같은 값이면 0으로 나누지 않도록 1로 둠
    public IReadOnlyList<double> Scale => _scale;

    public void Fit(IReadOnlyList<CustomerFeatures> rows)
  {
      if (rows.Count == 0)
    {
        throw new ArgumentException("학습할 고객이 없음", nameof(rows));
    }

    // count: 피처 개수 (20)
    // mean[k], scale[k]: k번째 피처의 평균과 표준편차
    int count = FeatureNames.All.Count;
    var mean = new double[count];
    var scale = new double[count];

    // 피처 하나씩 돌면서 학습 고객 전체의 평균과 표준편차를 구함
    for (int k = 0 ; k < count ; k++)
    {
        // 평균 = 모든 고객의 k번째 값을 더해 고객 수로 나눔
        double sum = 0.0;
        foreach (var c in rows)
      {
          sum += c.Values[k];
      }
      mean[k] = sum / rows.Count;

      // 편차의 제곱을 모두 더해 n으로 나눈 뒤 제곱근을 씌워 표준편차를 구함
      double sumSquares = 0.0;
      foreach (var c in rows)
      {
          double d = c.Values[k] - mean[k];
          sumSquares += d * d;
      }
      scale[k] = Math.Sqrt(sumSquares / rows.Count);

      // 모든 고객이 같은 값이면 표준편차가 0 이라 나눗셈이 불가능함
      // 1로 두면 (값 - 평균) / 1 = 0 이 되어 문제가 없음
      if (scale[k] == 0.0)
      {
          scale[k] = 1.0;
      }
    }

    // 끝까지 계산한 뒤에 저장함
    // 도중에 실패해도 반쯤 채워진 상태가 남지 않도록 함
    _mean = mean;
    _scale = scale;
  }

  // 고객 한 명의 피처 20개를 표준화한 값을 구함
  public double[] Transform(CustomerFeatures customer)
  {
      if (!IsFitted)
    {
        throw new InvalidOperationException("Fit 을 먼저 호출해야 함");
    }

    var z = new double[_mean.Length];
    for (int k = 0 ; k < z.Length ; k++)
    {
        // 표준화: (값 - 평균) / 표준편차
        // 예: 평균 5, 표준편차 2 일 때 값 9 는 (9 - 5) / 2 = 2
        z[k] = (customer.Values[k] - _mean[k]) / _scale[k];
    }
    return z;
  }
}