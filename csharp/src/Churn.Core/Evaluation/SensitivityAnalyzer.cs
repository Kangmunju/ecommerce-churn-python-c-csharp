namespace Churn.Core.Evaluation;


// 쿠폰을 보내라고 권할지에 대한 결론
// Skip: 보내면 손해이기 때문에 보내지 말 것
// BreakEven: 이익이 있어도 너무 작음
// Run: 이익이 충분해서 실행
public enum Advice
{
    Skip,
    BreakEven,
    Run,
}


// 가정 한 조합의 결과
// Best: 이 가정에서 이익이 가장 큰 임계값의 결과
public readonly record struct SensitivityRow(double Response, double CouponCost, ProfitPoint Best, Advice Advice);


// 가정이 틀렸다면 결론이 어떻게 바뀌는지 확인함
// 반응률과 쿠폰 원가는 데이터로 알 수 없는 가정
// 여러 값으로 바꾸어 가며 최적 임계값과 이익을 다시 구하도록 함
public static class SensitivityAnalyzer
{
    // 시험 고객 1,484명 기준으로 이익이 해당 값(파운드) 미만인지 확인함
    // 미만일 경우 운영 비용을 생각해 사실상 본전으로 봄 (가정)
    public const double DefaultMinWorth = 100.0;

    // baseline: 기준 가정, 여기서 반응률과 쿠폰 원가만 바꿔 가며 계산함
    // retainedRevenue: 유지 고객의 예측창 매출 중앙값
    // 반응률마다 쿠폰 원가를 모두 짝지어 돌려줌
    // responses 3개, couponCosts 4개면 12줄이 됨
    public static List<SensitivityRow> Analyze(
        IReadOnlyList<bool> churn,
        IReadOnlyList<double> probabilities,
        double retainedRevenue,
        ProfitAssumptions baseline,
        IReadOnlyList<double> responses,
        IReadOnlyList<double> couponCosts,
        double minWorth = DefaultMinWorth
    )
    {
        var rows = new List<SensitivityRow>();
        foreach (double response in responses)
        {
            foreach (double couponCost in couponCosts)
            {
                // with: 기준 가정을 복사하면서 적은 값만 바꿈
                var assumptions = baseline with { Response = response, CouponCost = couponCost };
                var values = ProfitCalculator.Values(assumptions, retainedRevenue);
                var curve = ProfitCalculator.Curve(churn, probabilities, values);
                var best = curve[ProfitCalculator.BestIndex(curve)];

                Advice advice;
                if (best.Profit <= 0.0)
                {
                    advice = Advice.Skip;
                }
                else if (best.Profit >= minWorth)
                {
                    advice = Advice.Run;
                }
                else
                {
                    advice = Advice.BreakEven;
                }
                rows.Add(new SensitivityRow(response, couponCost, best, advice));
            }
        }
        return rows;
    }
}


// ============================================================
// [설계 메모]
// ============================================================
// 1. 반응률과 쿠폰 원가만 바꾸는 이유
//    마진은 회계 값이라 비교적 알 수 있지만 반응률(쿠폰을 받고 남는 비율)은 실제로 보내 봐야 알 수 있는 가정임
//    기대이익이 이 두 값에 가장 크게 좌우되어서 바꿔 가며 결론이 뒤집히는지 확인함
//
// 2. 가정마다 최적 임계값을 다시 구함
//    반응률이 낮거나 쿠폰 원가가 높으면 최적 임계값이 0.95(거의 안 보냄)로 올라가고
//    반대면 0.05(거의 다 보냄)로 내려감 (파이썬 a5_sensitivity.csv 와 같은 결과)
//
// 3. 권고 기준 minWorth(100 파운드)는 가정임
//    이익이 0 이하면 Skip, 100 이상이면 Run, 그 사이는 운영 비용에 묻히는 수준이라 BreakEven 으로 봄
//    기준 값은 데이터에서 나온 것이 아니라 시험으로 바꿔 볼 수 있게 매개변수로 뺌
//
// 4. 한계
//    가정의 모든 조합(마진, 매출 중앙값 등)이 아니라 두 값만 바꿈
//    쿠폰 효과가 고객마다 다르다는 것은 반영하지 않고 반응률 하나로 평균 냄
