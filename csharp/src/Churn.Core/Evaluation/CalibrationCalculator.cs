namespace Churn.Core.Evaluation;


// 확률 구간 하나의 결과
// MeanPredicted: 이 구간 고객들의 예측 확률 평균
// ActualRate: 이 구간 고객들 중 실제 이탈한 비율
// Count: 이 구간의 고객 수
public readonly record struct CalibrationBin(double MeanPredicted, double ActualRate, int Count);


// 예측 확률이 진짜 확률처럼 맞는지 확인함
// 확률이 비슷한 고객끼리 같은 수로 묶음 (분위수 구간)
// 구간마다 예측 평균과 실제 이탈 비율을 비교함
// 두 값이 비슷하면 확률로 보고 기대이익을 계산하도록 함
// sklearn calibration_curve, strategy="quantile" 과 같은 방식
public static class CalibrationCalculator
{
    public static List<CalibrationBin> Compute(
        IReadOnlyList<bool> churn,
        IReadOnlyList<double> probabilities,
        int bins = 5
    )
    {
        if (churn.Count != probabilities.Count)
        {
            throw new ArgumentException($"정답 {churn.Count}개와 확률 {probabilities.Count}개의 개수가 다름");
        }
        if (probabilities.Count == 0)
        {
            throw new ArgumentException("고객이 없음", nameof(probabilities));
        }
        if (bins < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(bins), "구간은 1개 이상이어야 함");
        }

        // 구간의 경계 : 확률을 작은 순으로 세웠을 때 1/bins, 2/bins, ... 위치의 값
        // 5구간이면 경계는 20%, 40%, 60%, 80% 위치의 4개
        var sorted = probabilities.ToArray();
        Array.Sort(sorted);
        var edges = new double[bins - 1];
        for (int k = 1; k < bins; k++)
        {
            edges[k - 1] = Percentile(sorted, (double)k / bins);
        }

        // 고객마다 자기보다 작은 경계가 몇 개인지 세면 그것이 구간 번호(0 ~ bins-1)가 되도록 함
        var sumProbability = new double[bins];
        var sumChurn = new double[bins];
        var count = new int[bins];
        for (int i = 0; i < probabilities.Count; i++)
        {
            int bin = 0;
            while (bin < edges.Length && edges[bin] < probabilities[i])
            {
                bin++;
            }
            sumProbability[bin] += probabilities[i];
            if (churn[i])
            {
                sumChurn[bin] += 1.0;
            }
            count[bin]++;
        }

        // 고객이 한 명도 없는 구간은 평균을 낼 수 없기 때문에 건너뛰도록 함
        // sklearn 도 동일함
        var result = new List<CalibrationBin>();
        for (int b = 0; b < bins; b++)
        {
            if (count[b] > 0)
            {
                result.Add(new CalibrationBin(sumProbability[b] / count[b], sumChurn[b] / count[b], count[b]));
            }
        }
        return result;
    }

    // 정렬된 값에서 fraction(0 ~ 1) 위치의 값을 구함
    // 두 값 사이에 있다면 비례해서 섞음
    // numpy percentile 기본 방식과 동일함
    // 예: 값 10개에서 0.2 위치 = 9 * 0.2 = 1.8 번째 -> 1번째 값과 2번째 값 사이를 0.8 만큼 간 값
    private static double Percentile(double[] sorted, double fraction)
    {
        double position = (sorted.Length - 1) * fraction;
        int low = (int)Math.Floor(position);
        int high = Math.Min(low + 1, sorted.Length - 1);
        double weight = position - low;
        // 낮은 값 + (높은 값 - 낮은 값) * 비율
        return sorted[low] + (sorted[high] - sorted[low]) * weight;
    }
}


// ============================================================
// [설계 메모]
// ============================================================
// 1. 같은 폭이 아니라 같은 인원으로 묶는 이유 (분위수 구간)
//    확률을 0.0~0.2, 0.2~0.4 처럼 같은 폭으로 나누면 고객이 몰린 구간과 거의 없는 구간이 생김
//    같은 인원으로 묶으면 구간마다 296~297명이라 실제 비율이 안정적임
//
// 2. 구간 번호를 "자기보다 작은 경계의 개수"로 정하는 이유
//    경계와 같은 값은 아래 구간에 들어가서 sklearn 의 구간 나누기와 같아짐
//    확률이 전부 같으면 경계도 모두 같아서 한 구간에 몰리고 나머지는 건너뜀
//
// 3. 확인만 하고 보정(Platt, isotonic)은 하지 않음
//    시험 고객에서 예측 평균과 실제 비율이 구간마다 0.05 안팎으로 비슷하면 확률로 기대이익을 계산해도 됨
//    보정 모델을 따로 만들면 임계값·기대이익을 다시 검증해야 해서 이 프로젝트 범위에서 제외함
