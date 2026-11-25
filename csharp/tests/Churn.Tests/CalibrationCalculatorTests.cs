using Churn.Core.Evaluation;


namespace Churn.Tests;


// CalibrationCalculator 의 시험
// 작은 데이터는 손으로 계산한 값임
// 실제 데이터는 파이썬 reports/a5_calibration.csv 와 비교함
public class CalibrationCalculatorTests
{
    // 확률 0.1 ~ 1.0 열 명을 5구간으로 나눔
    // 구간 경계는 20%, 40%, 60%, 80% 위치로 설정함
    // 9 * 0.2 = 1.8 번째 -> 0.2 와 0.3 사이를 0.8 만큼 간 0.28
    // 같은 식으로 0.46, 0.64, 0.82 를 계산함

    // 구간 번호 = 자기보다 작은 경계의 개수
    // 0.1, 0.2 는 0번 / 0.3, 0.4 는 1번 / 0.5, 0.6 은 2번 / 0.7, 0.8 은 3번 / 0.9, 1.0 은 4번
    // 구간마다 2명이고 평균 확률은 0.15, 0.35, 0.55, 0.75, 0.95
    // 이탈은 4번째(0.4), 6, 7, 8, 9, 10 번째 고객
    // 따라서 구간별 실제 비율은 0, 0.5, 0.5, 1, 1 이 됨
    [Fact]
    public void Compute_matches_hand_calculation()
    {
        var probabilities = new List<double> { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
        var churn = new List<bool> { false, false, false, true, false, true, true, true, true, true };

        var bins = CalibrationCalculator.Compute(churn, probabilities);

        Assert.Equal(5, bins.Count);
        double[] meanPredicted = { 0.15, 0.35, 0.55, 0.75, 0.95 };
        double[] actualRate = { 0.0, 0.5, 0.5, 1.0, 1.0 };
        for (int b = 0; b < 5; b++)
        {
            Assert.Equal(2, bins[b].Count);
            Assert.Equal(meanPredicted[b], bins[b].MeanPredicted, 12);
            Assert.Equal(actualRate[b], bins[b].ActualRate, 12);
        }
    }

    // 확률이 전부 같으면 경계도 전부 같은 값이라 모두 0번 구간에 들어가는 것을 확인함
    // 고객이 없는 구간은 건너뛰어서 한 구간만 남게 됨
    [Fact]
    public void Compute_skips_empty_bins()
    {
        var probabilities = new List<double> { 0.5, 0.5, 0.5, 0.5 };
        var churn = new List<bool> { true, false, false, false };

        var bins = CalibrationCalculator.Compute(churn, probabilities);

        Assert.Single(bins);
        Assert.Equal(4, bins[0].Count);
        Assert.Equal(0.5, bins[0].MeanPredicted, 12);
        Assert.Equal(0.25, bins[0].ActualRate, 12);
    }

    [Fact]
    public void Compute_rejects_bad_input()
    {
        Assert.Throws<ArgumentException>(() => CalibrationCalculator.Compute(new List<bool> { true }, new List<double>()));
        Assert.Throws<ArgumentException>(() => CalibrationCalculator.Compute(new List<bool>(), new List<double>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => CalibrationCalculator.Compute(new List<bool> { true }, new List<double> { 0.5 }, 0));
    }

    // 정답 확률(reference_test_pred.csv)로 5구간을 나누면 파이썬 reports/a5_calibration.csv 와 같음
    // 파이썬 파일은 소수 셋째 자리까지만 적혀 있어서 셋째 자리까지 비교함
    // 시험 고객 1,484명이라 구간마다 296 ~ 297명
    [DataFact]
    public void Compute_matches_python_on_reference_predictions()
    {
        var (churn, probabilities) = TestData.ReadReferencePredictions();

        var bins = CalibrationCalculator.Compute(churn, probabilities);

        double[] meanPredicted = { 0.086, 0.313, 0.511, 0.633, 0.832 };
        double[] actualRate = { 0.135, 0.296, 0.527, 0.633, 0.808 };
        Assert.Equal(5, bins.Count);
        for (int b = 0; b < 5; b++)
        {
            Assert.Equal(meanPredicted[b], bins[b].MeanPredicted, 3);
            Assert.Equal(actualRate[b], bins[b].ActualRate, 3);
            Assert.InRange(bins[b].Count, 296, 297);
        }
        Assert.Equal(1484, bins.Sum(x => x.Count));
    }
}
