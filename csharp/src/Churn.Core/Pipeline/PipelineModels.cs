using Churn.Core.Evaluation;
using Churn.Core.Validation;


namespace Churn.Core.Pipeline;


// 파이프라인을 돌리기 위한 설정

// ExportDir: tools/export_for_c.py 가 만든 CSV 가 있는 폴더
// clean_transactions.csv, returns.csv, reference_split.csv 가 해당함

// Cutoff: 관측창 끝 = 예측창 시작
// LabelEnd: 예측창 끝
// Assumptions: 기대이익의 세 가지 가정 (마진, 쿠폰 원가, 반응률)
// MlpSeedCount: MLP 를 시드 0 ~ (개수 - 1) 로 학습해 보는 횟수
public sealed record PipelineOptions(
    string ExportDir,
    DateTime Cutoff,
    DateTime LabelEnd,
    ProfitAssumptions Assumptions,
    int MlpSeedCount
)
{
    // 파이썬과 동일한 기준일(2011-06-10), 예측창 끝(2011-12-10), 기본 가정, 시드 10개
    public static PipelineOptions ForDirectory(string exportDir)
    {
        return new PipelineOptions(exportDir, new DateTime(2011, 6, 10), new DateTime(2011, 12, 10), ProfitAssumptions.Default, 10);
    }
}


// 단계 하나가 걸린 시간 (초)
public sealed record StageTime(string Name, double Seconds);


// 파이프라인을 끝까지 돌린 결과
// Logistic ~ AtHalf: 로지스틱 모델을 시험 고객 1,484명에 적용한 결과
// OofThreshold, OofApplied: 학습 고객의 OOF 확률로 고른 임계값과 그것을 시험 고객에 적용한 결과
// LogisticCv, MlpCv: 전체 고객 5-fold 교차검증
// MlpSeedAucs: 시드마다 학습한 MLP 의 시험 AUC
public sealed record PipelineResult(
    int Customers,
    int Churned,
    int NewCustomers,
    int TrainCount,
    int TestCount,
    int LogisticIterations,
    Metrics Logistic,
    double RetainedMedian,
    IReadOnlyList<ProfitPoint> Curve,
    ProfitPoint Best,
    ProfitPoint AtHalf,
    double OofThreshold,
    ProfitPoint OofApplied,
    IReadOnlyList<CalibrationBin> Calibration,
    IReadOnlyList<SensitivityRow> Sensitivity,
    CrossValidationResult LogisticCv,
    CrossValidationResult MlpCv,
    IReadOnlyList<double> MlpSeedAucs,
    IReadOnlyList<StageTime> Times
);
