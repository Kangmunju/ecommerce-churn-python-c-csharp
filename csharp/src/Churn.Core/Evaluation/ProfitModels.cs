namespace Churn.Core.Evaluation;


// 쿠폰 발송에 대한 가정, 실무에서는 재무팀, 마케팅팀에서 받는 숫자
// Margin: 공헌 이익률 (매출 중 이익으로 남는 비율)
// CouponCost: 쿠폰 1장의 원가
// Response: 이탈 예정 고객이 쿠폰을 받고 돌아올 확률
public sealed record ProfitAssumptions(double Margin, double CouponCost, double Response)
{

    // 파이썬, C 와 동일한 가정을 세움
    public static readonly ProfitAssumptions Default = new(0.30, 5.00, 0.10);
}

// 쿠폰 1장을 보냈을 때의 이익
// 아무것도 보내지 않은 고객(FN, TN)은 이익, 손실 모두 0으로 체크
// ValueTp: 진짜 이탈할 고객에게 보냈을 경우
// ValueFp: 어차피 남을 고객에게 보냈을 경우
public readonly record struct ProfitValues(double ValueTp, double ValueFp);

// 유지 고객 한 명이 예측창에서 쓴 금액에 대한 실측값을 계산함
public readonly record struct RetainedRevenue(double Median, double Mean, int Count);

// 임계값 하나에 대한 결과
// Sent: 쿠폰을 보낸 고객 수 (Tp + Fp)
// Precision: 보낸 고객 중 진짜 이탈 비율 (보낸 고객이 없으면 0으로 체크함)
// Threshold: 이 임계값 이상의 확률을 가진 고객에게 쿠폰을 보냄
// Tp: 보낸 고객 중 실제 이탈, Fp: 보낸 고객 중 실제 유지
// Profit: Tp * ValueTp + Fp * ValueFp
public readonly record struct ProfitPoint(double Threshold, long Sent, long Tp, long Fp, double Precision, double Profit);
