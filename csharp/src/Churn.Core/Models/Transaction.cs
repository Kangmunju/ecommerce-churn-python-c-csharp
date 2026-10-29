namespace Churn.Core.Models;


// 거래 한 건 (clean_transactions.csv 의 한 행)
// 속성 순서는 CSV 열 순서와 같음
// record struct: 값만 담는 작은 묶음이기 때문에 구조체로 둠
//                한 번 만들면 바뀌지 않도록 readonly 로 작성함
public readonly record struct Transaction(
    int CustomerId,

    // 송장, 상품 코드는 계산에 쓰지 않고 글자가 섞여 있어서 string 으로 둠
    string Invoice,
    string StockCode,

    // C 에서는 1970년부터 지난 초로 직접 계산했던 값
    DateTime InvoiceDate,
    int Quantity,
    double Price,
    string Country
)
{
    // 이 줄의 매출 = 수량 * 가격
    // 저장하지 않고 읽을 때마다 계산함
    public double Revenue => Quantity * Price;
}
