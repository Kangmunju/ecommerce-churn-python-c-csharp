namespace Churn.Core.Evaluation;


// 혼동행렬: 실제와 예측을 짝지어 센 네 가지 개수
// Tn: 실제 유지, 예측 유지 (맞게 유지로 봄)
// Fp: 실제 유지, 예측 이탈 (헛되이 이탈로 봄)
// Fn: 실제 이탈, 예측 유지 (이탈을 놓침)
// Tp: 실제 이탈, 예측 이탈 (맞게 이탈로 봄)
// 순서는 파이썬 confusion_matrix 의 [[TN, FP], [FN, TP]] 와 동일함
public readonly record struct ConfusionMatrix(long Tn, long Fp, long Fn, long Tp);

// 평가 지표 모음
// Auc, PrAuc, Brier 는 확률 자체로 구해 임계값의 영향을 받지 않음
// Accuracy, Precision, Recall, F1, Matrix 는 임계값 이상을 이탈로 본 결과
public readonly record struct Metrics(
    double Auc,
    double PrAuc,
    double Accuracy,
    double Precision,
    double Recall,
    double F1,
    double Brier,
    ConfusionMatrix Matrix
);