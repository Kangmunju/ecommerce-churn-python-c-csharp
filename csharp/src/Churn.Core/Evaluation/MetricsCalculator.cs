namespace Churn.Core.Evaluation;


// 정답(이탈 여부)과 예측 확률로 평가 지표를 계산함
// 모델 종류와 상관없이 정답과 확률만 받으므로 로지스틱 회귀, MLP, 교차검증 결과에 모두 쓸 수 있음
public static class MetricsCalculator
{
    
    // churn[i]: i번째 고객의 정답
    // probabilities[i]: i번째 고객의 이탈 확률
    public static Metrics Evaluate(IReadOnlyList<bool> churn, IReadOnlyList<double> probabilities, double threshold = 0.5)
  {
      if (churn.Count != probabilities.Count)
    {
        throw new ArgumentException($"정답 {churn.Count}개와 확률 {probabilities.Count}개의 개수가 다름");
    }
    if (churn.Count == 0)
    {
        throw new ArgumentException("평가할 고객이 없음");
    }

    int n = churn.Count;
    long tn = 0;
    long fp = 0;
    long fn = 0;
    long tp = 0;
    double brierSum = 0.0;

    // 확률을 기준으로 정렬하면서 정답이 같이 따라다니도록 (확률, 정답) 쌍으로 묶음
    // 튜플의 이름 Probability, Churn 으로 scored[i].Probability 처럼 꺼내 쓸 수 있음
    var scored = new List<(double Probability, bool Churn)>(n);

    // 혼동행렬, Brier 점수, 정렬용 목록을 한 번에 만듦
    for (int i = 0 ; i < n ; i++)
    {
        bool predicted = probabilities[i] >= threshold;

        if (churn[i])
      {
          if (predicted)
        {
            tp++;
        }
        else
        {
            fn++;
        }
      }
      else
      {
          if (predicted)
        {
            fp++;
        }
        else
        {
            tn++;
        }
      }

      // Brier: 확률과 정답(이탈 1, 유지 0)의 차이를 제곱해서 평균을 구함
      // 작을수록 좋음
      double answer = 0.0;
      if (churn[i])
      {
          answer = 1.0;
      }
      double diff = probabilities[i] - answer;
      brierSum += diff * diff;

      scored.Add((probabilities[i], churn[i]));
    }

    // 확률이 높은 쪽이 앞에 오도록 정렬함
    // (a, b) => b 를 a 와 비교하면 큰 쪽이 앞에 오는 내림차순
    scored.Sort((a, b) => b.Probability.CompareTo(a.Probability));

    // tp + fn: 실제 이탈 고객 전체, tn + fp: 실제 유지 고객 전체
    var (auc, prAuc) = AucAndPrAuc(scored, tp + fn, tn + fp);

    // 정밀도: 이탈로 예측한 고객 중 실제 이탈한 비율
    // 재현율: 실제 이탈한 고객 중 잡아낸 비율
    double precision = SafeDivide(tp, tp + fp);
    double recall = SafeDivide(tp, tp + fn);

    return new Metrics(
        Auc: auc,
        PrAuc: prAuc,
        Accuracy: (double)(tp + tn) / n,
        Precision: precision,
        Recall: recall,
        F1: SafeDivide(2.0 * precision * recall, precision + recall),
        Brier: brierSum / n,
        Matrix: new ConfusionMatrix(tn, fp, fn, tp)
    );
  }

  // 분모가 0 이면 0 을 돌려줌 (sklearn 의 zero_division=0 과 같음)
  private static double SafeDivide(double numerator, double denominator)
  {
      if (denominator == 0.0)
    {
        return 0.0;
    }
    return numerator / denominator;
  }

  // 확률이 높은 순으로 정렬된 목록을 훑으며 AUC 와 PR-AUC 를 함께 구함
  // 같은 확률의 고객들은 한 묶음으로 보고 묶음 단위로 계산함
  // totalPos: 전체 이탈 고객 수
  // totalNeg: 전체 유지 고객 수
  // 돌려주는 값 (Auc, PrAuc) 는 이름 붙은 튜플이라 var (auc, prAuc) = 처럼 한 번에 받을 수 있음
  private static (double Auc, double PrAuc) AucAndPrAuc(List<(double Probability, bool Churn)> sorted, long totalPos, long totalNeg)
  {
      // seenPos, seenNeg: 앞선 묶음들(더 높은 확률)에 있던 이탈, 유지 고객 수
      long seenPos = 0;
      long seenNeg = 0;

      // pairScore: 이탈 고객과 유지 고객을 하나씩 짝지었을 때 이탈 고객의 확률이 더 높은 쌍의 수
      // 확률이 같은 쌍은 0.5 쌍으로 셈
      // 마지막에 전체 쌍의 수(이탈 고객 수 * 유지 고객 수)로 나누면 AUC
      double pairScore = 0.0;

      // ap: 재현율이 늘어난 만큼에 그때의 정밀도를 곱해 더한 값(PR-AUC)
      // prevRecall: 바로 앞 묶음까지의 재현율, 이번 묶음에서 재현율이 얼마나 늘었는지 알기 위해 기억함
      double ap = 0.0;
      double prevRecall = 0.0;

      int i = 0;
      while (i < sorted.Count)
    {

        // 확률이 sorted[i] 와 같은 고객들이 한 묶음
        int j = i;
        long groupPos = 0;
        long groupNeg = 0;
        while (j < sorted.Count && sorted[j].Probability == sorted[i].Probability)
      {
          if (sorted[j].Churn)
        {
            groupPos++;
        }
        else
        {
            groupNeg++;
        }
        j++;
      }

      // 이 묶음의 이탈 고객이 이기는 유지 고객 수
      // 확률이 더 낮은 유지 고객(전체 - 앞선 묶음 - 이 묶음)은 1쌍씩
      // 같은 묶음의 유지 고객은 0.5쌍씩
      long lowerNeg = totalNeg - seenNeg - groupNeg;
      pairScore += groupPos * (lowerNeg + 0.5 * groupNeg);

      // 이 묶음까지를 이탈로 예측했을 때의 정밀도와 재현율
      seenPos += groupPos;
      seenNeg += groupNeg;
      double precision = SafeDivide(seenPos, seenPos + seenNeg);
      double recall = SafeDivide(seenPos, totalPos);

      ap += (recall - prevRecall) * precision;
      prevRecall = recall;

      i = j;
    }

    return (SafeDivide(pairScore, (double)totalPos * totalNeg), ap);
  }
}


// ====================================================================================================
// [설계 메모]
// ====================================================================================================
// 1. AUC 를 정렬 한 번으로 구하는 이유
//    정의대로 하면 이탈 고객 x 유지 고객의 모든 쌍을 비교해야 해서 시험 고객 1,484명 기준 약 55만 번 비교함
//    확률이 높은 순으로 한 번 정렬하고 앞에서부터 훑으면 이탈 고객마다 "뒤에 있는 유지 고객 수" 만 세면 됨
//    정렬 비용(n log n) 만 들고, 고객이 수십만 명으로 늘어도 같은 방식으로 계산 가능함
//    PR-AUC 도 같은 정렬 결과를 훑으면서 함께 구하므로 정렬은 한 번만 함
//
// 2. 같은 확률은 한 묶음으로 처리함
//    확률이 정확히 같은 고객들 사이에 순서를 매기면 정렬 방식에 따라 AUC 가 달라짐
//    (List.Sort 는 같은 값의 순서를 지키지 않음)
//    그래서 같은 확률의 고객은 묶음으로 보고, 묶음 안의 이탈 x 유지 쌍은 0.5 쌍씩 셈 (sklearn 과 같은 규칙)
//    확률을 전부 0.5 로 준 시험에서 AUC 가 정확히 0.5, PR-AUC 가 이탈률로 나오는지 확인함
//
// 3. 혼동행렬 칸의 순서와 임계값
//    TN, FP, FN, TP 순서이며 파이썬 confusion_matrix 의 [[TN, FP], [FN, TP]] 와 같음
//    정답 확률로 계산하면 TN 527, FP 245, FN 181, TP 531 이고 C# 이 학습한 모델의 확률로 계산하면 526, 246, 180, 532 로 1명씩 다름
//    정답 CSV 의 계수가 sklearn 이 일찍 멈춘 값이라 확률이 최대 8e-3 달랐고, 0.5 근처에 있던 고객이 반대편으로 넘어간 것임
//    (LogisticRegression.cs 하단 설계 메모 4 참고, C 의 결과와 같음)
//    AUC, PR-AUC, Brier 는 확률 자체로 계산해 임계값의 영향을 받지 않고, 나머지 지표는 임계값 이상을 이탈로 봄
//
// 4. C 와 달라진 점
//    C 는 int 배열과 double 배열을 따로 받고 Scored 구조체와 qsort 를 썼음
//    C# 은 IReadOnlyList<bool>, IReadOnlyList<double> 을 받고 이름 붙은 튜플과 List.Sort 를 씀
//    정답과 확률만 받으므로 로지스틱 회귀 외에 MLP, 교차검증 결과(OOF 확률)에도 그대로 쓸 수 있음
// ====================================================================================================