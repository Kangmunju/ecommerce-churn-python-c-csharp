namespace Churn.Core.Models;


// 고객 한 명의 이탈 라벨과 관측창 요약
// FirstPurchase: 관측창에서 처음으로 산 시각
// LastPurchase: 관측창에서 마지막으로 산 시각
// OrderCount: 관측창에서 서로 다른 송장 수
// Churn: true = 예측창에 구매 없음(이탈), false = 예측창에 구매 있음(유지)
// IsNew: true = 관측창 마지막 30일 안에 첫 구매
// C 에서는 Churn, IsNew 를 int(0 또는 1)로 두었지만 여기서는 bool 로 둠
public readonly record struct CustomerLabel(
    int CustomerId,
    DateTime FirstPurchase,
    DateTime LastPurchase,
    int OrderCount,
    bool Churn,
    bool IsNew
);
