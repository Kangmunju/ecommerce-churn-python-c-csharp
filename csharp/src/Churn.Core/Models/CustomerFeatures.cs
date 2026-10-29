namespace Churn.Core.Models;


// 고객 한 명의 피처 20개와 정답
// record 의 같은 비교는 배열을 '내용'이 아닌 '주소'로 비교함
// 값이 동일한 두 고객도 다르다고 인식하는 문제점이 있음
// 문제 해결을 위해 class 를 사용함
public sealed class CustomerFeatures
{

    // 20개 피처 값
    // Feature 번호 순서
    private readonly double[] _values;

    public int CustomerId { get; }

    // true = 이탈
    public bool Churn {get;}

    // true = 관측창 마지막 30일 안에 첫 구매
    public bool IsNew {get;}

    public CustomerFeatures(int customerId, double[] values, bool churn, bool isNew)
  {
      // 피처가 20개가 아니면 만들지 않고 오류를 냄
      if (values.Length != FeatureNames.All.Count)
    {
        throw new ArgumentException(
            $"피처는 {FeatureNames.All.Count}개여야 함 (받은 개수: {values.Length})", nameof(values)
        );
    }

    CustomerId = customerId;

    // 바깥 배열을 나중에 바꿔도 이 고객의 값이 바뀌지 않도록 복사해 둠
    _values = (double[])values.Clone();
    Churn = churn;
    IsNew = isNew;
  }


  // 이름으로 값 읽기
  public double this[Feature feature] => _values[(int)feature];

  // 20개를 순서대로 돌 때 쓰는 읽기 전용 보기
  // 모델이 for문으로 돌게 하기 위해 작성함
  // 복사 없이 같은 배열을 읽기만 가능하게 보여 줌
  public ReadOnlySpan<double> Values => _values;
}
