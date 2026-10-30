using Churn.Core.Models;


namespace Churn.Core.Features;


// 거래와 반품에서 고객별 피처 20개를 만듦
// 관측창(InvoiceDate < cutoff)의 기록만 사용함
// 거래 줄 -> 주문(같은 고객의 같은 송장) -> 고객 피처 순으로 접음
// C 에서는 목적마다 정렬을 다시 했던 부분을 Dictionary 와 HashSet 으로 해결함
public static class FeatureBuilder
{

    // 최근 활동을 보는 기간
    private const int RecentDays = 90;

    // recency_ratio 의 상한
    private const double RecencyRatioMax = 50.0;

    private const string UkName = "United Kingdom";

    // 주문 한 건은 같은 고객의 같은 송장의 거래 줄을 접은 것을 나타냄
    private sealed class Order
  {
      // 송장에서 가장 이른 시각 (주문 시각)
      public DateTime FirstTime;

      // 송장에서 가장 늦은 줄의 시각
      public DateTime LastTime;

      // 수량 * 가격의 합
      public double Value;

      // 수량의 합
      public double Items;

      // 가장 늦은 줄의 국가가 영국이면 True를 리턴함
      public bool LastIsUk;
  }

  // labels 와 같은 순서로 리턴함
  // 고객 번호 오름차순
  public static List<CustomerFeatures> Build(
      IEnumerable<Transaction> transactions,
      IEnumerable<ReturnSlip> returns,
      IReadOnlyList<CustomerLabel> labels,
      DateTime cutoff
  )
  {
      // (고객, 송장) 한 쌍으로 주문을 찾음
      // 튜플은 값이 같으면 같은 키로 분류함
      var orderOf = new Dictionary<(int, string), Order>();
      var ordersByCustomer = new Dictionary<int, List<Order>>();
      var productsByCustomer = new Dictionary<int, HashSet<string>>();

      foreach (var t in transactions)
    {
        if (t.InvoiceDate >= cutoff)
      {
          continue;
      }

      bool isUk = t.Country == UkName;

      if (!orderOf.TryGetValue((t.CustomerId, t.Invoice), out var order))
      {
          order = new Order { FirstTime = t.InvoiceDate, LastTime = t.InvoiceDate, LastIsUk = isUk };
          orderOf.Add((t.CustomerId, t.Invoice), order);

          if (!ordersByCustomer.TryGetValue(t.CustomerId, out var orders))
        {
            orders = new List<Order>();
            ordersByCustomer.Add(t.CustomerId, orders);
            productsByCustomer.Add(t.CustomerId, new HashSet<string>());
        }
        orders.Add(order);
      }

      order.Value += t.Revenue;
      order.Items += t.Quantity;
      if (t.InvoiceDate < order.FirstTime)
      {
          order.FirstTime = t.InvoiceDate;
      }
      if (t.InvoiceDate > order.LastTime)
      {
          order.LastTime = t.InvoiceDate;
          order.LastIsUk = isUk;
      }

      productsByCustomer[t.CustomerId].Add(t.StockCode);
    }

    // 고객별로 취소 송장을 모음
    // 같은 송장은 한 번만 세어지도록 함
    var returnsByCustomer = new Dictionary<int, HashSet<string>>();
    foreach (var r in returns)
    {
        if (r.InvoiceDate >= cutoff)
      {
          continue;
      }

      if (!returnsByCustomer.TryGetValue(r.CustomerId, out var invoices))
      {
          invoices = new HashSet<string>();
          returnsByCustomer.Add(r.CustomerId, invoices);
      }
      invoices.Add(r.Invoice);
    }

    var result = new List<CustomerFeatures>(labels.Count);

    foreach (var label in labels)
    {
        if (!ordersByCustomer.TryGetValue(label.CustomerId, out var orders))
      {
          throw new InvalidOperationException($"고객 {label.CustomerId} 의 관측창 거래가 없음");
      }

      // OrderBy 는 같은 시각이면 들어온 순서를 지킴
      // cf) List.Sort 는 지키지 않음
      var sorted = orders.OrderBy(o => o.FirstTime).ToList();
      var x = FromOrders(sorted, cutoff);

      int nReturns = 0;
      if (returnsByCustomer.TryGetValue(label.CustomerId, out var returnInvoices))
      {
          nReturns = returnInvoices.Count;
      }

      x[(int)Feature.NProducts] = productsByCustomer[label.CustomerId].Count;
      x[(int)Feature.NReturns] = nReturns;
      x[(int)Feature.ReturnRate] = Math.Min(nReturns / x[(int)Feature.Frequency], 1.0);

      result.Add(new CustomerFeatures(label.CustomerId, x, label.Churn, label.IsNew));
    }

    return result;
  }

  // 고객 한 명의 주문 목록(주문 시각 순)에서 반품, 상품 수를 뺀 나머지 피처를 채움
  private static double[] FromOrders(List<Order> orders, DateTime cutoff)
  {
      int n = orders.Count;
      var recentStart = cutoff.AddDays(-RecentDays);

      double monetary = 0.0;
      double items = 0.0;
      double maxValue = orders[0].Value;
      double orders90 = 0.0;
      double revenue90 = 0.0;
      var latestRowTime = orders[0].LastTime;
      bool isUk = orders[0].LastIsUk;

      foreach (var o in orders)
    {
        monetary += o.Value;
        items += o.Items;
        if (o.Value > maxValue)
      {
          maxValue = o.Value;
      }
      if (o.FirstTime >= recentStart)
      {
          orders90 += 1.0;
          revenue90 += o.Value;
      }

      // 가장 늦은 줄이 속한 주문의 국가를 씀
      // 같은 시각이면 뒤의 주문을 쓰도록 함
      if (o.LastTime >= latestRowTime)
      {
          latestRowTime = o.LastTime;
          isUk = o.LastIsUk;
      }
    }

    var orderValues = new List<double>(n);
    foreach (var o in orders)
    {
        orderValues.Add(o.Value);
    }
    double meanValue = monetary / n;

    // 날짜 차이는 일수만 셈
    // TimeSpan.Days 는 시, 분, 초를 버림
    var first = orders[0].FirstTime;
    var last = orders[n - 1].FirstTime;
    double recency = (cutoff - last).Days;
    double tenure = (cutoff - first).Days;

    // 주문이 1번이면 간격이 없으므로 지금까지 기다린 기간(tenure)으로 채움
    double gapMean = tenure;
    double gapStd = 0.0;
    double gapMax = tenure;
    if (n > 1)
    {
        var gaps = new List<double>(n - 1);
        for (int i = 1 ; i < n ; i++)
      {
          gaps.Add((orders[i].FirstTime - orders[i - 1].FirstTime).Days);
      }
      gapMean = gaps.Average();
      gapStd = SampleStd(gaps);
      gapMax = gaps.Max();
    }

    // 간격 평균이 0이면 나누지 않고 1.0으로 표기함
    double recencyRatio = 1.0;
    if (gapMean != 0.0)
    {
        recencyRatio = recency / gapMean;
    }
    if (recencyRatio > RecencyRatioMax)
    {
        recencyRatio = RecencyRatioMax;
    }

    var x = new double[FeatureNames.All.Count];
        x[(int)Feature.Recency] = recency;
        x[(int)Feature.Frequency] = n;
        x[(int)Feature.Monetary] = monetary;
        x[(int)Feature.AvgOrderValue] = meanValue;
        x[(int)Feature.StdOrderValue] = SampleStd(orderValues);
        x[(int)Feature.MaxOrderValue] = maxValue;
        x[(int)Feature.TotalItems] = items;
        x[(int)Feature.Tenure] = tenure;
        x[(int)Feature.ActiveSpan] = (last - first).Days;
        x[(int)Feature.GapMean] = gapMean;
        x[(int)Feature.GapStd] = gapStd;
        x[(int)Feature.GapMax] = gapMax;
        x[(int)Feature.RecencyRatio] = recencyRatio;
        x[(int)Feature.Orders90d] = orders90;
        x[(int)Feature.Revenue90d] = revenue90;
        x[(int)Feature.Share90d] = revenue90 / monetary;
        double isUkValue = 0.0;
        if (isUk)
        {
            isUkValue = 1.0;
        }
        x[(int)Feature.IsUk] = isUkValue;
        return x;
  }

  // 표본 표준편차
  // n - 1로 나눔
  // pandas 기본값
  // 값이 1개 이하면 0을 반환함
  private static double SampleStd(List<double> values)
  {
      if (values.Count < 2)
    {
        return 0.0;
    }

    double mean = values.Average();
    double sumSquares = 0.0;
    foreach (var v in values)
    {
        double d = v - mean;
        sumSquares += d * d;
    }
    return Math.Sqrt(sumSquares / (values.Count - 1));
  }
}