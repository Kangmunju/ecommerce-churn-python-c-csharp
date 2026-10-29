namespace Churn.Core.Models;


// 피처 번호
// reference_features.csv 의 열 순서와 같음
// features[Feature.Recency]를 이용해 숫자 대신 이름으로 부름
// enum 은 이름에 0부터 번호가 자동으로 붙음 (Recency = 0, ... IsUk = 19)
// 각 피처의 정의는 파이썬 src/features.py 참고
public enum Feature
{
    Recency,
    Frequency,
    Monetary,
    AvgOrderValue,
    StdOrderValue,
    MaxOrderValue,
    TotalItems,
    Tenure,
    ActiveSpan,
    GapMean,
    GapStd,
    GapMax,
    RecencyRatio,
    Orders90d,
    Revenue90d,
    Share90d,
    NProducts,
    NReturns,
    ReturnRate,
    IsUk,
}


// CSV 머리글과 같은 글자로 피처 이름을 작성함
// 번호 순서가 Feature 와 같도록 주의함 (어긋나면 ModelTests 가 실패함)
public static class FeatureNames
{
    public static readonly IReadOnlyList<string> All =
    [
        "recency", "frequency", "monetary", "avg_order_value", "std_order_value",
        "max_order_value", "total_items", "tenure", "active_span",
        "gap_mean", "gap_std", "gap_max", "recency_ratio",
        "orders_90d", "revenue_90d", "share_90d",
        "n_products", "n_returns", "return_rate", "is_uk"
    ];
}
