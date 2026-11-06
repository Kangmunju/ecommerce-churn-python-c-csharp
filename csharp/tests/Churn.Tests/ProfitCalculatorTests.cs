using Churn.Core.Data;
using Churn.Core.Evaluation;
using Churn.Core.Labels;
using Churn.Core.Models;
using Churn.Core.Modeling;


namespace Churn.Tests;


// ProfitCalculator 의 시험
// 작은 데이터는 손으로 계산한 값
// 실제 데이터는 tools/expected_values.py 의 출력과 비교할 것
public class ProfitCalculatorTests
{
    private static readonly DateTime Cutoff = new(2011, 6, 10);
    private static readonly DateTime LabelEnd = new(2011, 12, 10);

    // 시험용 거래 한 줄, 고객, 시각, 수량, 가격만 의미 있음
    private static Transaction Line(int customer, DateTime time, int quantity, double price)
  {
      return new Transaction(customer, "A1", "S1", time, quantity, price, "United Kingdom");
  }

  // 시험용 라벨, 고객 번호와 이탈 여부만 의미 있음
  private static CustomerLabel Label(int customer, bool churn)
  {
      return new CustomerLabel(customer, new DateTime(2010, 1, 1), new DateTime(2011, 5, 1), 1, churn, false);
  }

  // 쿠폰 1장의 이익 = 반응률 * 매출 * 마진 - 쿠폰 원가
  // 0.10 * 635.01 * 0.30 - 5 = 14.0503
  // 보내지 않아도 될 고객에게 보낸 경우 쿠폰 원가 5 만 잃는 것
  [Fact]
  public void Values_computes_profit_per_coupon()
  {
      var values = ProfitCalculator.Values(ProfitAssumptions.Default, 635.01);

      Assert.Equal(14.0503, values.ValueTp, 10);
      Assert.Equal(-5.0, values.ValueFp);
  }

  // 유지 고객 3명의 예측창 매출을 구함
  // 고객 4 는 이탈에 해당해 제외함
  // 고객 1: 2 * 10 + 1 * 5 = 25 (관측창의 거래 1000 은 제외)
  // 고객 2: 100 (labelEnd 와 같은 시각의 거래, 999 는 예측창 밖이라 제외함)
  // 고객 3: 40 (Cutoff 와 같은 시각의 거래는 예측창에 포함됨)
  // 정렬하면 25 40 100, 중앙값 40, 평균 55
  // Line(고객, 시각, 수량, 가격) 의 매출은 수량 * 가격
  [Fact]
  public void MeasureRetained_computes_median_with_odd_count()
  {
      var transactions = new List<Transaction>
      {
          Line(1, new DateTime(2011, 5, 1), 1, 1000.0),
          Line(1, new DateTime(2011, 7, 1), 2, 10.0),
          Line(1, new DateTime(2011, 8, 1), 1, 5.0),
          Line(2, new DateTime(2011, 9, 1), 1, 100.0),
          Line(2, LabelEnd, 1, 999.0),
          Line(3, Cutoff, 1, 40.0),
      };
      var labels = new List<CustomerLabel> { Label(1, false), Label(2, false), Label(3, false), Label(4, true) };

      var r = ProfitCalculator.MeasureRetained(transactions, labels, Cutoff, LabelEnd);

      Assert.Equal(40.0, r.Median);
      Assert.Equal(55.0, r.Mean, 12);
      Assert.Equal(3, r.Count);
  }

  // 짝수 명이면 가운데 두 값의 평균을 중앙값으로 사용함
  // 10 25 40 100 -> (25 + 40) / 2 = 32.5
  // 평균 43.75
  [Fact]
  public void MeasureRetained_averages_the_middle_two_with_even_count()
  {
      var transactions = new List<Transaction>
      {
          Line(1, new DateTime(2011, 7, 1), 1, 25.0),
          Line(2, new DateTime(2011, 7, 1), 1, 100.0),
          Line(3, new DateTime(2011, 7, 1), 1, 40.0),
          Line(5, new DateTime(2011, 7, 1), 1, 10.0),
      };
      var labels = new List<CustomerLabel> { Label(1, false), Label(2, false), Label(3, false), Label(5, false) };

      var r = ProfitCalculator.MeasureRetained(transactions, labels, Cutoff, LabelEnd);

      Assert.Equal(32.5, r.Median);
      Assert.Equal(43.75, r.Mean);
      Assert.Equal(4, r.Count);
  }

  [Fact]
  public void MeasureRetained_rejects_when_nobody_stayed()
  {
      var labels = new List<CustomerLabel> { Label(1, true) };

      Assert.Throws<InvalidOperationException>(() => ProfitCalculator.MeasureRetained(new List<Transaction>(), labels, Cutoff, LabelEnd));
  }

  // 고객 4명: 확률 0.9(이탈), 0.6(유지), 0.3(이탈), 0.1(유지)
  // 맞게 보낸 경우 +10 으로 계산
  // 보내지 않아도 되는데 보낸 경우 -5 으로 계산
  // 0.05 ~ 0.10: 4명 모두 발송, Tp 2 Fp 2 -> 20 - 10 = 10
  // 0.15 ~ 0.30: 0.9 0.6 0.3 발송, Tp 2 Fp 1 -> 20 - 5 = 15
  // 0.35 ~ 0.60: 0.9 0.6 발송, Tp 1 Fp 1 -> 10 - 5 = 5
  // 0.65 ~ 0.90: 0.9 만 발송, Tp 1 -> 10
  // 0.95: 아무도 안 보냄 -> 0
  // 임계값과 같은 확률(0.1, 0.3, 0.6, 0.9)은 발송에 포함됨
  [Fact]
  public void Curve_counts_sent_customers_for_each_threshold()
  {
      var churn = new[] { true, false, true, false };
      var probability = new[] { 0.9, 0.6, 0.3, 0.1 };

      var curve = ProfitCalculator.Curve(churn, probability, new ProfitValues(10.0, -5.0));

      // curve[k] 의 임계값은 0.05 * (k + 1)
      // curve[0] = 0.05, curve[1] = 0.10, curve[5] = 0.30, curve[9] = 0.50
      // curve[11] = 0.60, curve[17] = 0.90, curve[18] = 0.95
      Assert.Equal(ProfitCalculator.ThresholdCount, curve.Count);
      Assert.Equal(0.05, curve[0].Threshold);
      Assert.Equal(0.95, curve[18].Threshold);

      // ProfitPoint(임계값, Sent, Tp, Fp, Precision, Profit), 정밀도 = Tp / Sent = 2 / 4
      Assert.Equal(new ProfitPoint(0.05, 4, 2, 2, 0.5, 10.0), curve[0]);

      // 0.10: 확률 0.1 이 임계값과 같아서 발송에 포함됨
      Assert.Equal(4, curve[1].Sent);

      Assert.Equal(new ProfitPoint(0.30, 3, 2, 1, 2.0 / 3.0, 15.0), curve[5]);
      Assert.Equal(new ProfitPoint(0.60, 2, 1, 1, 0.5, 5.0), curve[11]);
      Assert.Equal(new ProfitPoint(0.90, 1, 1, 0, 1.0, 10.0), curve[17]);

      // 아무도 안 보내면 정밀도 0, 이익 0
      Assert.Equal(new ProfitPoint(0.95, 0, 0, 0, 0.0, 0.0), curve[18]);
  }

  // 이익이 가장 큰 임계값을 고름
  // 값이 동일한 경우에는 앞쪽의 값을 선택함
  // 위 데이터에서 0.15 ~ 0.30 이 모두 15 이므로 0.15(위치 2)를 고름
  [Fact]
  public void BestIndex_picks_the_first_maximum()
  {
      var churn = new[] { true, false, true, false };
      var probability = new[] { 0.9, 0.6, 0.3, 0.1 };
      var curve = ProfitCalculator.Curve(churn, probability, new ProfitValues(10.0, -5.0));

      int best = ProfitCalculator.BestIndex(curve);

      Assert.Equal(2, best);
      Assert.Equal(0.15, curve[best].Threshold);
  }

  [Fact]
  public void Curve_rejects_mismatched_input()
  {
      Assert.Throws<ArgumentException>(() =>
            ProfitCalculator.Curve(new[] { true }, new[] { 0.5, 0.6 }, new ProfitValues(10.0, -5.0)));
  }

  // 유지 고객 2,572명의 예측창 매출
  // tools/expected_values.py 의 [유지 고객의 예측창 매출] 과 동일한 값
  // 중앙값 635.01, 평균 1758.29 로 크게 다른 결과가 나왔음
  // 큰손 몇 명이 평균을 끌어올린 것
  [DataFact]
  public void MeasureRetained_matches_python_on_real_data()
  {
      var transactions = CsvLoader.LoadTransactions(TestData.Transactions);
      var labels = LabelBuilder.Build(transactions, Cutoff, LabelEnd);

      var r = ProfitCalculator.MeasureRetained(transactions, labels, Cutoff, LabelEnd);

      Assert.Equal(2572, r.Count);
      Assert.Equal(635.01, r.Median, 8);
      Assert.Equal(1758.2904315708, r.Mean, 8);
  }

  // 정답 확률로 임계값 19개 곡선을 그리면 파이썬과 동일해야 함
  // 기대값은 tools/expected_values.py 의 [임계값별 이익 (reference_test_pred.csv)]
  [DataFact]
  public void Curve_matches_python_on_reference_predictions()
  {
      var (churn, probability) = TestData.ReadReferencePredictions();
      var values = ProfitCalculator.Values(ProfitAssumptions.Default, 635.01);

      var curve = ProfitCalculator.Curve(churn, probability, values);
      int best = ProfitCalculator.BestIndex(curve);

      Assert.Equal(0.30, curve[best].Threshold);
      Assert.Equal(1056, curve[best].Sent);
      Assert.Equal(643, curve[best].Tp);
      Assert.Equal(413, curve[best].Fp);
      Assert.Equal(6969.3429, curve[best].Profit, 4);

      // 임계값 0.50 은 curve[9]
      Assert.Equal(776, curve[9].Sent);
      Assert.Equal(6235.7093, curve[9].Profit, 4);

      // 19개를 모두 더한 값으로 곡선 전체가 맞는지 확인함
      Assert.Equal(13380, curve.Sum(p => p.Sent));

      // Assert.Equal(기대, 실제, 자릿수) 는 반올림해서 비교함
      // 반올림 경계에 걸리면 갈리게 됨
      // 차이의 크기를 직접 비교할 것
      // 람다의 p 는 곡선의 한 점(ProfitPoint), 바깥의 probability 와 이름이 겹치지 않게 p 로 씀
      Assert.True(Math.Abs(curve.Sum(p => p.Profit) - 91884.2505) < 1e-6);
  }

  // C# 이 학습한 모델의 확률로도 최적 임계값은 0.30, 이익은 £6,969.34 가 됨
  // 임계값 0.50(curve[9]) 은 정답 CSV 가 일찍 멈춘 값이었음
  // 정답 확률일 때 £6,235.71 이었고 C# 모델은 C 와 같은 £6,244.76 으로 약간 다름
  // 635.01 은 유지 고객 매출 중앙값 (위 MeasureRetained 시험의 값)
  [DataFact]
  public void Curve_gives_same_result_as_c_for_trained_model()
  {
      var model = new LogisticRegression();
      model.Fit(TestData.Train);
      var churn = new List<bool>();
      var probability = new List<double>();
      foreach (var c in TestData.Test)
    {
        churn.Add(c.Churn);
        probability.Add(model.PredictProbability(c));
    }
    var values = ProfitCalculator.Values(ProfitAssumptions.Default, 635.01);

    var curve = ProfitCalculator.Curve(churn, probability, values);
    int best = ProfitCalculator.BestIndex(curve);

    Assert.Equal(0.30, curve[best].Threshold);
    Assert.Equal(6969.3429, curve[best].Profit, 4);
    Assert.Equal(6244.7596, curve[9].Profit, 4);
  }

}