using Churn.Core.Models;


namespace Churn.Core.Labels;


// 거래 목록에서 고객별 이탈 라벨을 만듦
// 관측창: InvoiceDate < cutoff
// 예측창: cutoff <= InvoiceDate < labelEnd
// 관측창에 1건 이상 구매한 고객만 대상이 됨
// C 에서는 정렬 후 bsearch 로 찾았던 부분을 고객 번호로 바로 찾는 구조인 Dictionary로 해결함
public static class LabelBuilder
{

    // 신규: 관측창 끝 30일 안에 첫 구매한 고객
    private const int NewCustomerDays = 30;

    // 관측창에서 고객 한 명의 요약을 쌓는 임시 묶음을 생성함
    private sealed class Summary
    {
        public DateTime First;
        public DateTime Last;

        // HashSet: 같은 값을 한 번만 담는 모음
        // 같은 송장의 여러 줄이 주문 1건으로 세어지도록 작성함
        public readonly HashSet<string> Invoice = new();
    }

    // 고객 번호를 오름차순으로 돌려줌
    public static List<CustomerLabel> Build(IEnumerable<Transaction> transactions, DateTime cutoff, DateTime labelEnd)
    {
        if (labelEnd <= cutoff)
        {
            throw new ArgumentException("labelEnd 는 cutoff 보다 뒤여야 함");
        }

        // Dictionary<키, 값>: 고객 번호(키)로 요약(값)을 바로 찾도록 작성함
        var summaries = new Dictionary<int, Summary>();
        var futureBuyers = new HashSet<int>();

        foreach (var t in transactions)
        {
            if (t.InvoiceDate < cutoff)
            {
                // TryGetValue: 있으면 true 와 함께 out 변수에 값을 담아 줌
                if (!summaries.TryGetValue(t.CustomerId, out var summary))
                {
                    summary = new Summary { First = t.InvoiceDate, Last = t.InvoiceDate };
                    summaries.Add(t.CustomerId, summary);
                }

                if (t.InvoiceDate < summary.First)
                {
                    summary.First = t.InvoiceDate;
                }
                if (t.InvoiceDate > summary.Last)
                {
                    summary.Last = t.InvoiceDate;
                }
                summary.Invoice.Add(t.Invoice);
            }
            else if (t.InvoiceDate < labelEnd)
            {
                futureBuyers.Add(t.CustomerId);
            }
        }

        var newLimit = cutoff.AddDays(-NewCustomerDays);
        var labels = new List<CustomerLabel>(summaries.Count);

        // foreach (var (a, b) in 사전) 문법을 이용해 키와 값을 각각 a, b로 풀어서 받도록 작성함
        foreach (var (customerId, summary) in summaries)
        {
            labels.Add(new CustomerLabel(
                CustomerId: customerId,
                FirstPurchase: summary.First,
                LastPurchase: summary.Last,
                OrderCount: summary.Invoice.Count,
                Churn: !futureBuyers.Contains(customerId),
                IsNew: summary.First >= newLimit
            ));
        }

        // Dictionary 는 순서를 보장하지 않으므로 고객 번호 순으로 정렬함
        labels.Sort((a, b) => a.CustomerId.CompareTo(b.CustomerId));
        return labels;
    }
}
