using Churn.Core.Models;


namespace Churn.Core.Evaluation;


// 임계값별 쿠폰 발송의 기대이익을 계산함
// 쿠폰 1장의 이익은 유지 고객의 예측창 매출 중앙값과 3 가지 가정으로 정하였음
// 3 가지 가정은 마진, 쿠폰 원가, 반응률에 관한 것
public static class ProfitCalculator
{
    
    // 임계값 개수: 0.05 부터 0.95 까지 0.05 간격으로 설정함
    public const int ThresholdCount = 19;

    // 관측창과 예측창 모두 구매한 고객을 유지 고객으로 설정함
    // 유지 고객이 예측창에서 쓴 금액의 중앙값, 평균, 인원을 계산함
    // C 에서는 labels 에서 bsearch 로 고객 위치를 찾았던 부분을 Dictionary 로 해결함
    public static RetainedRevenue MeasureRetained(
        IEnumerable<Transaction> transactions,
        IReadOnlyList<CustomerLabel> labels,
        DateTime cutoff,
        DateTime labelEnd
    )
  {
      // 예측창의 거래를 훑으며 고객별 금액의 합을 쌓음
      // 금액은 수량 * 가격으로 계산함
      var revenueOf = new Dictionary<int, double>();
      foreach (var t in transactions)
    {
        if (t.InvoiceDate < cutoff || t.InvoiceDate >= labelEnd)
      {
          continue;
      }

      revenueOf.TryGetValue(t.CustomerId, out double current);
      revenueOf[t.CustomerId] = current + t.Revenue;
    }

    // 예측창에 구매한 고객(Churn == false)의 금액만 모음
    var kept = new List<double>();
    double sum = 0.0;
    foreach (var label in labels)
    {
        if (label.Churn)
      {
          continue;
      }

      revenueOf.TryGetValue(label.CustomerId, out double revenue);
      kept.Add(revenue);
      sum += revenue;
    }

    if (kept.Count == 0)
    {
        throw new InvalidOperationException("예측창에 구매한 고객이 없음");
    }

    // 정렬 후 가운데 값을 중앙값으로 사용함
    // 짝수 명인 경우에는 가운데 두 값의 평균을 사용함
    kept.Sort();
    int middle = kept.Count / 2;
    double median;
    if (kept.Count % 2 == 1)
    {
        median = kept[middle];
    }
    else
    {
        median = (kept[middle - 1] + kept[middle]) / 2.0;
    }

    return new RetainedRevenue(median, sum / kept.Count, kept.Count);
  }

  // 가정과 유지 고객의 매출(중앙값)로 쿠폰 1장의 이익을 구함
  // 맞게 보낸 경우: 반응률 * 매출 * 마진 - 쿠폰 원가
  // 보내지 않아도 되는데 보낸 경우: -쿠폰 원가
  // retainedRevenue: MeasureRetained 가 돌려준 값의 Median (평균이 아니라 중앙값을 넣음)
  public static ProfitValues Values(ProfitAssumptions assumptions, double retainedRevenue)
  {
      double valueTp = assumptions.Response * retainedRevenue * assumptions.Margin - assumptions.CouponCost;
      double valueFp = -assumptions.CouponCost;
      return new ProfitValues(valueTp, valueFp);
  }

  // 임계값을 0.05 부터 0.95 까지 바꾸어 가며 그 임계값으로 쿠폰을 보냈다면 얼마가 남는지를 계산함
  // 임계값을 낮추면 더 많이 보내 진짜 이탈자를 더 잡을 수 있음 (TP 증가)
  // 임계값을 낮추면 보낼 필요가 없는데 보내는 경우도 늘어남 (FP 증가)
  // 돌려주는 목록의 k번째 칸이 임계값 0.05 * (k + 1) 의 결과 (0번째 0.05, 9번째 0.50, 18번째 0.95)
  public static List<ProfitPoint> Curve(IReadOnlyList<bool> churn , IReadOnlyList<double> probabilities, ProfitValues values)
  {
      if (churn.Count != probabilities.Count)
    {
        throw new ArgumentException($"정답 {churn.Count}개와 확률 {probabilities.Count}개의 개수가 다름");
    }

    var curve = new List<ProfitPoint>(ThresholdCount);

    for (int k = 0 ;k < ThresholdCount ; k++)
    {

        // 0.05 를 계속 더하다보면 오차가 쌓이는 문제를 확인함
        // 5씩 늘리는 정수(5, 10, ... 95)를 100으로 나누어 매번 정확하게 계산함
        double threshold = 5 * (k + 1) / 100.0;

        // 확률이 임계값 이상이면 쿠폰을 보냄
        // 그 중 정답이 이탈이면 Tp, 유지인 경우에는 Fp
        // 보내지 않은 고객은 이익, 손실 아무 것도 없기 때문에 세지 않음
        long tp = 0;
        long fp = 0;
        for (int i = 0 ; i < churn.Count ; i++)
      {
          if (probabilities[i] >= threshold)
        {
            if (churn[i])
          {
              tp++;
          }
          else
          {
            fp++;
          }
        }
      }

      // 한 명도 보내지 않은 경우는 나눌 수 없으므로 정밀도를 0으로 둠
      double precision = 0.0;
      if (tp + fp > 0)
      {
          precision = (double)tp / (tp + fp);
      }

      // 기대 이익 = 맞게 보낸 수 * 맞게 보낸 1건의 이익 + 헛되이 보낸 수 * 헛되이 보낸 1건의 손실
      // ValueFp 는 -5처럼 음수이기 때문에 더하면 손실이 빠지는 것과 동일한 결과가 나옴
      double profit = tp * values.ValueTp + fp * values.ValueFp;

      // 인자 순서: 임계값, 보낸 수(Sent = tp + fp), Tp, Fp, 정밀도, 이익
      curve.Add(new ProfitPoint(threshold, tp + fp, tp, fp, precision, profit));
    }

    return curve;
  }

  // 곡선에서 이익이 가장 큰 임계값의 위치를 구함
  // 같은 위치가 존재할 경우에는 앞쪽 값을 사용함
  public static int BestIndex(IReadOnlyList<ProfitPoint> curve)
  {
      int best = 0;
      for (int k = 1 ; k < curve.Count ; k++)
    {
        if (curve[k].Profit > curve[best].Profit)
      {
          best = k;
      }
    }
    return best;
  }
}


// ====================================================================================================
// [설계 메모]
// ====================================================================================================
// 1. 쿠폰 1장의 이익이 14.05 가 되는 이유
//    진짜 이탈할 고객에게 보내면 10% 확률로 돌아오고, 돌아오면 평균적으로 635.01 파운드를 쓰고, 그 중 30% 가 이익임
//    0.10 * 635.01 * 0.30 - 5 = 14.0503
//    어차피 남을 고객에게 보내면 쿠폰 값 5 파운드만 잃음, 보내지 않은 고객은 이익도 손실도 0
//    그래서 이익은 "맞게 보낸 수 * 14.05 - 헛되이 보낸 수 * 5" 한 줄로 정해짐
//
// 2. 평균 대신 중앙값을 쓰는 이유
//    유지 고객 2,572명의 예측창 매출은 중앙값 635.01, 평균 1758.29 로 크게 다름
//    큰손 몇 명이 평균을 끌어올리므로 평균을 쓰면 쿠폰의 이익을 과대평가하게 됨
//    한 명이 쓰는 금액으로는 중앙값이 더 보수적이고 현실에 가까움
//    매출은 가정이 아니라 데이터에서 잰 값이고, 마진, 쿠폰 원가, 반응률은 가정임 (ProfitAssumptions.Default 에 모아 둠)
//
// 3. 한계 (파이썬 a5_profit.py 에서 확인한 내용, 이 단계에서는 아직 구현하지 않음)
//    임계값을 시험 고객에서 골랐기 때문에 이 이익은 낙관적일 수 있음
//    파이썬은 학습 고객의 out-of-fold 확률(교차검증)에서 임계값을 고른 뒤 시험 고객에 적용해 확인했음 (0.25 에서 6,909 파운드)
//    교차검증은 8단계, OOF 임계값과 확률 보정 확인, 가정이 틀렸을 때의 민감도 분석은 10단계에서 다룸
//    쿠폰을 받아서 돌아온 것인지 원래 돌아올 사람인지는 관찰 데이터로 알 수 없어서 반응률은 가정으로 둘 수밖에 없음
//
// 4. C 와 달라진 점
//    C 는 예측창 거래를 훑으며 labels 에서 bsearch 로 고객 위치를 찾았고, 가정은 main.c 의 #define 이었음
//    C# 은 Dictionary<int, double> 로 고객별 매출을 바로 찾고, 가정은 ProfitAssumptions 한 곳에 모아 둠
// ====================================================================================================