using Churn.Core.Pipeline;


namespace Churn.Tests;


// PipelineRunner 의 시험
// 파이프라인을 한 번 돌려 파이썬 정답지(reports/*.csv, reference_summary.csv)와 C 구현 결과에 맞는지 확인함
// 전체를 도는 데 10초 넘게 걸려서 한 번만 돌리고 시험들이 같은 결과를 씀
public class PipelineRunnerTests
{
    // Lazy: 처음 쓸 때 한 번만 실행할 것
    private static readonly Lazy<PipelineResult> Result = new(
        () => PipelineRunner.Run(PipelineOptions.ForDirectory(TestData.ExportDir))
    );

    // 고객 4,944명
    // 이탈 2,372명
    // 신규 105명
    // 학습 3,460명
    // 시험 1,484명
    [DataFact]
    public void Run_counts_match_python()
    {
        var r = Result.Value;

        Assert.Equal(4944, r.Customers);
        Assert.Equal(2372, r.Churned);
        Assert.Equal(105, r.NewCustomers);
        Assert.Equal(3460, r.TrainCount);
        Assert.Equal(1484, r.TestCount);
    }

    // 로지스틱: AUC 는 파이썬 0.781834 와 1e-3 이내
    // 혼동행렬: C 구현과 동일한 526 / 246 / 180 / 532
    // 정답 CSV 가 덜 수렴한 값이기 때문에 파이썬(527 / 245 / 181 / 531)과는 1명씩 옮겨 간 것임
    [DataFact]
    public void Run_logistic_matches_python_and_c()
    {
        var r = Result.Value;

        Assert.InRange(r.Logistic.Auc, 0.781834 - 1e-3, 0.781834 + 1e-3);
        Assert.Equal(526, r.Logistic.Matrix.Tn);
        Assert.Equal(246, r.Logistic.Matrix.Fp);
        Assert.Equal(180, r.Logistic.Matrix.Fn);
        Assert.Equal(532, r.Logistic.Matrix.Tp);
    }

    // 기대이익: 시험 고객에서 고른 최적 0.30 에 6,969.34 파운드, OOF 로 고른 임계값은 0.25
    [DataFact]
    public void Run_profit_matches_python()
    {
        var r = Result.Value;

        Assert.Equal(635.01, r.RetainedMedian, 2);
        Assert.Equal(0.30, r.Best.Threshold);
        Assert.InRange(r.Best.Profit, 6969.34 - 0.01, 6969.34 + 0.01);
        Assert.Equal(0.25, r.OofThreshold);
        Assert.Equal(19, r.Curve.Count);
        Assert.Equal(5, r.Calibration.Count);
        Assert.Equal(12, r.Sensitivity.Count);
    }

    // 교차검증과 MLP: 평균을 파이썬의 범위와 비교함
    // 로지스틱 CV 0.8044
    // MLP CV 0.8046
    // MLP 시드 10개 평균 0.7836 (표준편차 0.0035)
    // 조각과 난수가 sklearn 과 다르기 때문에 조각별 값이 아니라 평균만 범위로 볼 것
    [DataFact]
    public void Run_cross_validation_and_mlp_are_in_python_range()
    {
        var r = Result.Value;

        Assert.InRange(r.LogisticCv.MeanAuc, 0.8044 - 0.003, 0.8044 + 0.003);
        Assert.InRange(r.MlpCv.MeanAuc, 0.8046 - 0.005, 0.8046 + 0.005);
        Assert.Equal(10, r.MlpSeedAucs.Count);
        Assert.InRange(r.MlpSeedAucs.Average(), 0.7836 - 0.003, 0.7836 + 0.003);
    }

    // 단계마다 시간을 기록함
    [DataFact]
    public void Run_records_a_time_for_every_stage()
    {
        var r = Result.Value;

        Assert.Equal(9, r.Times.Count);
        Assert.All(r.Times, t => Assert.True(t.Seconds >= 0.0));
    }
}
