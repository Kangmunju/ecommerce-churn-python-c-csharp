namespace Churn.Core.Validation;

// 교차검증 결과
// FoldAucs: 조각마다 그 조각을 시험용으로 쓴 AUC
// MeanAuc, StdAuc: FoldAucs 의 평균과 표준편차 (n 으로 나눔, 파이썬 numpy 의 std 와 같음)
// OofProbabilities: 고객마다 그 고객이 들어있지 않은 조각들로 학습한 모델이 예측한 확률
//                   입력으로 넣은 고객 목록과 같은 순서
public sealed record CrossValidationResult(
    IReadOnlyList<double> FoldAucs,
    double MeanAuc,
    double StdAuc,
    IReadOnlyList<double> OofProbabilities
);
