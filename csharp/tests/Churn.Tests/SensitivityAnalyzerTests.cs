using Churn.Core.Evaluation;


namespace Churn.Tests;


// SensitivityAnalyzer 의 시험
// 작은 데이터는 손으로 계산한 값
// 실제 데이터는 파이썬 reports/a5_sensitivity.csv 와 비교함
public class SensitivityAnalyzerTests
{
    // 고객 4명: 정답 이탈, 이탈, 유지, 유지
    // 확률은 순서대로 0.9, 0.62, 0.56, 0.12
    private static readonly List<bool> Churn = new() { true, true, false, false };
    private static readonly List<double> Probabilities = new() { 0.9, 0.62, 0.56, 0.12 };

    // 마진 100%, 쿠폰 원가 1, 반응률 50%, 유지 고객 매출 10인 경우를 가정함
    // 맞게 보낸 1건 = 0.5 * 10 * 1.0 - 1 = 4
    // 헛되이 보낸 1건 = -1
    // 임계값 별 이익: 0.05 ~ 0.10 은 4명(2건 맞음, 2건 헛됨) 모두 보냈으므로 8 - 2 = 6
    //               0.15 ~ 0.55 는 3명(2건 맞음, 1건 헛됨)이므로 8 - 1 = 7
    //               0.60 은 2명(2건 맞음)이므로 8
    //               0.65 ~ 0.90 은 1명이므로 4
    //               0.95 는 0명이므로 0
    // 따라서 구한 최적은 0.60, 이익 8
    // 쿠폰 원가가 20 이면 맞게 보낸 1건 = 5 - 20 = -15, 헛되이 보낸 1건 = -20
    // 보낼수록 손해라는 결론에 도달함
    // 아무도 보내지 않는 0.95 의 이익 0 이 가장 커서 권고는 하지 않는 것으로 함
    [Fact]
    public void Analyze_gives_advice_by_profit()
    {
        var baseline = new ProfitAssumptions(Margin: 1.0, CouponCost: 1.0, Response: 0.5);
        var responses = new[] { 0.5 };
        var couponCosts = new[] { 1.0, 20.0 };

        // 기본 기준 100 이면 이익 8은 너무 적어 사실상 본전으로 판단됨
        var strict = SensitivityAnalyzer.Analyze(Churn, Probabilities, 10.0, baseline, responses, couponCosts);

        // 기준을 5 로 낮추면 이익 8이 실행되는 것을 확인함
        var loose = SensitivityAnalyzer.Analyze(Churn, Probabilities, 10.0, baseline, responses, couponCosts, minWorth: 5.0);

        Assert.Equal(0.60, strict[0].Best.Threshold);
        Assert.Equal(2, strict[0].Best.Sent);
        Assert.Equal(8.0, strict[0].Best.Profit, 12);
        Assert.Equal(Advice.BreakEven, strict[0].Advice);
        Assert.Equal(Advice.Run, loose[0].Advice);

        Assert.Equal(0.95, strict[1].Best.Threshold);
        Assert.Equal(0, strict[1].Best.Sent);
        Assert.Equal(Advice.Skip, strict[1].Advice);
    }

    // 반응률마다 쿠폰 원가를 모두 짝지어서 돌려줌
    // 반응률이 바깥 반복임에 주의할 것
    [Fact]
    public void Analyze_returns_every_combination_in_order()
    {
        var baseline = ProfitAssumptions.Default;

        var rows = SensitivityAnalyzer.Analyze(Churn, Probabilities, 10.0, baseline, new[] { 0.5, 0.1 }, new[] { 1.0, 20.0 });

        Assert.Equal(4, rows.Count);
        Assert.Equal((0.5, 1.0), (rows[0].Response, rows[0].CouponCost));
        Assert.Equal((0.5, 20.0), (rows[1].Response, rows[1].CouponCost));
        Assert.Equal((0.1, 1.0), (rows[2].Response, rows[2].CouponCost));
        Assert.Equal((0.1, 20.0), (rows[3].Response, rows[3].CouponCost));
    }

    // 정답 확률로 반응률 4개 * 쿠폰 원가 3개 = 12줄을 생성함
    // 파이썬 reports/a5_sensitivity.csv 와 동일함
    // 파이썬 파일의 기대이익은 정수로 반올림되어 있어서 0.5 이내로 비교함
    // 유지 고객 매출 중앙값은 635.01
    [DataFact]
    public void Analyze_matches_python_on_reference_predictions()
    {
        var (churn, probabilities) = TestData.ReadReferencePredictions();

        var rows = SensitivityAnalyzer.Analyze(churn, probabilities, 635.01, ProfitAssumptions.Default, new[] { 0.03, 0.05, 0.10, 0.20 }, new[] { 2.0, 5.0, 15.0 });

        // 반응률, 쿠폰 원가, 최적 임계값, 발송, 기대이익(반올림), 권고
        var expected = new (double, double, double, long, double, Advice)[]
        {
            (0.03, 2.0, 0.30, 1056, 1563, Advice.Run),
            (0.03, 5.0, 0.95, 22, 10, Advice.BreakEven),
            (0.03, 15.0, 0.95, 22, -210, Advice.Skip),
            (0.05, 2.0, 0.20, 1181, 4039, Advice.Run),
            (0.05, 5.0, 0.50, 776, 1178, Advice.Run),
            (0.05, 15.0, 0.95, 22, -130, Advice.Skip),
            (0.10, 2.0, 0.05, 1374, 10682, Advice.Run),
            (0.10, 5.0, 0.30, 1056, 6969, Advice.Run),
            (0.10, 15.0, 0.80, 169, 170, Advice.Run),
            (0.20, 2.0, 0.05, 1374, 24113, Advice.Run),
            (0.20, 5.0, 0.05, 1374, 19991, Advice.Run),
            (0.20, 15.0, 0.45, 875, 8935, Advice.Run),
        };
        Assert.Equal(12, rows.Count);
        for (int k = 0; k < expected.Length; k++)
        {
            var (response, coupon, threshold, sent, profit, advice) = expected[k];
            Assert.Equal(response, rows[k].Response);
            Assert.Equal(coupon, rows[k].CouponCost);
            Assert.Equal(threshold, rows[k].Best.Threshold);
            Assert.Equal(sent, rows[k].Best.Sent);
            Assert.InRange(rows[k].Best.Profit, profit - 0.5, profit + 0.5);
            Assert.Equal(advice, rows[k].Advice);
        }
    }
}
