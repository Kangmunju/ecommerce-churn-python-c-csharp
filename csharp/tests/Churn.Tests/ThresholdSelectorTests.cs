using Churn.Core.Evaluation;
using Churn.Core.Modeling;
using Churn.Core.Validation;


namespace Churn.Tests;


// ThresholdSelector 의 시험
// 작은 데이터는 손으로 계산한 값임
// 실제 데이터는 파이썬 a5_profit.py 의 OOF 임계값 검증과 비교함
public class ThresholdSelectorTests
{
    // SensitivityAnalyzerTests 와 같은 고객 4명, 맞게 보낸 1건 = 4, 헛되이 보낸 1건 = -1
    // 임계값 0.60 에서 이익 8 이 최대 (2명에게 보내 2건 맞음)
    private static readonly List<bool> Churn = new() { true, true, false, false };
    private static readonly List<double> Probabilities = new() { 0.9, 0.62, 0.56, 0.12 };
    private static readonly ProfitValues Values = new(4.0, -1.0);

    [Fact]
    public void ChooseFromOof_picks_the_most_profitable_threshold()
    {
        double threshold = ThresholdSelector.ChooseFromOof(Churn, Probabilities, Values);

        Assert.Equal(0.60, threshold);
    }

    // 임계값 0.60 을 다른 고객 2명에게 적용함
    // 확률 0.7 (이탈), 0.65 (유지) 둘 다 0.60 이상이기 때문에 2명에게 보냄
    // 1건이 맞고 1건이 헛됨: 4 - 1 = 3
    [Fact]
    public void Apply_gives_the_result_for_the_chosen_threshold()
    {
        var churn = new List<bool> { true, false };
        var probabilities = new List<double> { 0.7, 0.65 };

        var point = ThresholdSelector.Apply(churn, probabilities, Values, 0.60);

        Assert.Equal(2, point.Sent);
        Assert.Equal(1, point.Tp);
        Assert.Equal(1, point.Fp);
        Assert.Equal(3.0, point.Profit, 12);
    }

    // 곡선에 없는 임계값(0.07 은 0.05 간격이 아님)은 거절하도록 함
    [Fact]
    public void Apply_rejects_a_threshold_not_on_the_curve()
    {
        Assert.Throws<ArgumentException>(() => ThresholdSelector.Apply(Churn, Probabilities, Values, 0.07));
    }

    // 학습 고객 3,460명의 OOF 확률(5-fold, 시드 42)로 임계값을 고르면 파이썬과 같이 0.25 가 됨
    // 이 임계값을 시험 고객의 정답 확률(reference_test_pred.csv)에 적용하면 1,110명에게 보내 기대이익 6,908.8962
    // 시험 고객에서 고른 최적 0.30 의 6,969.34 보다 60 파운드 낮고 기본값 0.50 의 6,235.71 보다 673 파운드 높음
    // 시험셋에서 임계값을 고르면 낙관적이라는 것을 보여 주는 숫자
    [DataFact]
    public void OofThreshold_matches_python_on_real_data()
    {
        var values = ProfitCalculator.Values(ProfitAssumptions.Default, 635.01);
        var trainChurn = TestData.Train.Select(c => c.Churn).ToList();
        var oof = CrossValidator.Run(TestData.Train, () => new LogisticRegression(), 5, 42).OofProbabilities;
        var (churn, probabilities) = TestData.ReadReferencePredictions();

        double threshold = ThresholdSelector.ChooseFromOof(trainChurn, oof, values);
        var applied = ThresholdSelector.Apply(churn, probabilities, values, threshold);

        Assert.Equal(0.25, threshold);
        Assert.Equal(1110, applied.Sent);
        Assert.True(Math.Abs(applied.Profit - 6908.8962) < 1e-3, $"기대이익 {applied.Profit}");
    }

    // 같은 과정을 C# 으로 학습한 모델의 확률로 함
    // 정답 CSV 가 덜 수렴한 값이기 때문에 시험 고객 확률이 조금 다름
    // 그래서 1,111명에게 보내 기대이익 6,922.9465 가 됨
    [DataFact]
    public void OofThreshold_works_with_the_trained_model()
    {
        var values = ProfitCalculator.Values(ProfitAssumptions.Default, 635.01);
        var trainChurn = TestData.Train.Select(c => c.Churn).ToList();
        var oof = CrossValidator.Run(TestData.Train, () => new LogisticRegression(), 5, 42).OofProbabilities;
        var model = new LogisticRegression();
        model.Fit(TestData.Train);
        var testChurn = TestData.Test.Select(c => c.Churn).ToList();
        var testProbabilities = TestData.Test.Select(c => model.PredictProbability(c)).ToList();

        double threshold = ThresholdSelector.ChooseFromOof(trainChurn, oof, values);
        var applied = ThresholdSelector.Apply(testChurn, testProbabilities, values, threshold);

        Assert.Equal(0.25, threshold);
        Assert.Equal(1111, applied.Sent);
        Assert.True(Math.Abs(applied.Profit - 6922.9465) < 1e-3, $"기대이익 {applied.Profit}");
    }
}
