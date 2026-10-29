namespace Churn.Core.Models;


// 반품(취소) 전표 한 건 (returns.csv 의 한 행)
// 송장 번호는 맨 앞에 C가 붙음 ex. C489449
// 반품은 건수만 세므로 수량, 가격은 없음
public readonly record struct ReturnSlip(
    int CustomerId,
    string Invoice,
    DateTime InvoiceDate
);
