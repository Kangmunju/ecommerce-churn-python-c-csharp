namespace Churn.Core.Evaluation;


// 쿠폰 임계값을 학습 고객의 OOF 확률로 고르고 다른 고객들에게 적용해 봄
// 시험 고객에서 임계값을 고르면 그 고객들에게 맞춘 값이라 이익이 낙관적으로 나옴
// OOF 확률은 모델이 처음 보는 고객에 대한 예측이기 때문에 임계값을 고를 때 공정함
public static class ThresholdSelector
{
    // 이익이 가장 큰 임계값을 돌려줌
    // 같은 이익이면 앞쪽 값을 선택하도록 함
    // churn, oofProbabilities: 학습 고객의 정답과 OOF 확률, 같은 순서여야 함
    public static double ChooseFromOof(
        IReadOnlyList<bool> churn,
        IReadOnlyList<double> oofProbabilities,
        ProfitValues values
    )
    {
        var curve = ProfitCalculator.Curve(churn, oofProbabilities, values);
        return curve[ProfitCalculator.BestIndex(curve)].Threshold;
    }

    // 고른 임계값을 다른 고객들(시험 고객)에게 적용했을 때의 결과를 돌려줌
    public static ProfitPoint Apply(
        IReadOnlyList<bool> churn,
        IReadOnlyList<double> probabilities,
        ProfitValues values,
        double threshold
    )
    {
        var curve = ProfitCalculator.Curve(churn, probabilities, values);
        foreach (var point in curve)
        {
            // 소수는 == 로 비교하면 끝자리 오차에 걸릴 수 있음
            // 아주 작은 차이는 같은 값으로 보도록 설정함
            if (Math.Abs(point.Threshold - threshold) < 1e-9)
            {
                return point;
            }
        }
        throw new ArgumentException($"임계값 {threshold} 은 곡선에 없음 (0.05 ~ 0.95, 0.05 간격)", nameof(threshold));
    }
}


// ============================================================
// [설계 메모]
// ============================================================
// 1. 임계값을 학습 고객의 OOF 확률로 고르는 이유
//    7단계처럼 시험 고객에서 최적 임계값(0.30, 6,969.34 파운드)을 고르면 시험 고객에게 맞춘 값이라 낙관적임
//    OOF 확률로 고르면 시험 고객을 전혀 보지 않고 고른 값이라 실제 운영과 같은 조건이 됨
//    결과는 0.25, 시험 고객에 적용하면 6,908.90 파운드로 0.30 보다 60 파운드 낮음 (이 차이가 낙관의 크기)
//
// 2. 곡선에 없는 임계값은 예외로 거절함
//    0.07 처럼 0.05 간격이 아닌 값은 곡선에서 찾을 수 없어서 조용히 가까운 값을 쓰지 않고 알림
//
// 3. 한계
//    임계값 하나만 OOF 로 골랐을 뿐 반응률·쿠폰 원가 같은 가정은 그대로임
//    가정이 틀렸을 때 결론은 SensitivityAnalyzer 가 확인함
