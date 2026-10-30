using System.Globalization;
using Churn.Core.Data;
using Churn.Core.Labels;
using Churn.Core.Models;


namespace Churn.Tests;


// LabelBuilder 의 시험
// 기대값은 tools/expected_values.py 의 출력
// C 시험과 동일한 값
public class LabelBuilderTests
{
    // Cutoff: 관측창과 예측창을 가르는 기준 시각
    // 거래 시각이 Cutoff 미만이면 관측창, 이상이면 예측창으로 판정함
    // LabelEnd: 예측창의 끝 시각 (미만까지만 예측창)
    private static readonly DateTime Cutoff = new(2011, 6, 10);
    private static readonly DateTime LabelEnd = new(2011, 12, 10);

    // 시험용 거래 한 건
    // 고객, 송장, 시각만 의미 있음
    // 송장 이름은 시험용으로 지음
    // 글자 = 고객 (A는 1번, B는 2번 ...)
    // 숫자 = 그 고객의 몇 번째 송장
    private static Transaction Tx(int customer, string invoice, DateTime time)
  {
      return new Transaction(customer, invoice, "S1", time, 1, 1.0, "United Kingdom");
  }

  // 고객 4명을 번호가 섞인 순서로 넣고 각 경우를 확인함
  // 판정 기준
  // 1: 관측창에 송장 2개, 예측창에도 구매 -> 유지
  // 2: 관측창에만 구매 -> 이탈
  // 3: 예측창에만 구매 -> 대상 아님
  // 4: 관측창 마지막 30일 안에 첫 구매 -> 신규
  // new DateTime(년, 월, 일) 은 0시 0분 0초, 시 분 초까지 적으면 그 시각
  // 같은 송장의 두 줄 중 이른 시각(9시 0분)이 FirstPurchase 가 되는지 확인하기 위해 고객 1 의 A1 만 시각을 적었음
  [Fact]
  public void Build_labels_each_case()
  {
      var transactions = new List<Transaction>
      {
          Tx(4, "D1", new DateTime(2011, 6, 1)),
          Tx(2, "B1", new DateTime(2010, 1, 5)),
          Tx(1, "A1", new DateTime(2010, 3, 1, 9, 0, 0)),
          Tx(1, "A1", new DateTime(2010, 3, 1, 9, 5, 0)),
          Tx(1, "A2", new DateTime(2011, 5, 20)),
          Tx(1, "A3", new DateTime(2011, 8, 1)),
          Tx(3, "C1", new DateTime(2011, 7, 1)),
      };

      var labels = LabelBuilder.Build(transactions, Cutoff, LabelEnd);

      // 고객 3 은 빠지고 번호 순서로 정렬됨
      // labels.Select(l => l.CustomerId): 각 라벨에서 고객 번호만 뽑은 목록
      Assert.Equal(new[] { 1, 2, 4 }, labels.Select(l => l.CustomerId));

      var one = labels[0];
      Assert.Equal(2, one.OrderCount);
      Assert.Equal(new DateTime(2010, 3, 1, 9, 0, 0), one.FirstPurchase);
      Assert.Equal(new DateTime(2011, 5, 20), one.LastPurchase);
      Assert.False(one.Churn);
      Assert.False(one.IsNew);

      Assert.True(labels[1].Churn);
      Assert.False(labels[1].IsNew);

      Assert.True(labels[2].IsNew);
      Assert.True(labels[2].Churn);
  }

  // 경계 기준
  // cutoff 와 같은 시각은 예측창으로 정함
  // labelEnd 와 같은 시각은 예측창 밖으로 정함
  // AddDays(-30): 30일 전 시각을 새로 만들어 줌 (음수는 뺌, 원래 값은 바뀌지 않음)
  // AddSeconds(-1): 1초 전, 이어서 쓰면 앞의 결과에 다시 적용됨
  // -100 은 관측창 안이기만 하면 되는 아무 날짜를 가리킴
  // -30 은 LabelBuilder 의 NewCustomerDays 와 같아야 함
  [Fact]
  public void Build_boundaries()
  {
      var transactions = new List<Transaction>
      {
          Tx(1, "A1", Cutoff.AddDays(-100)),
          Tx(1, "A2", Cutoff),
          Tx(2, "B1", Cutoff.AddDays(-100)),
          Tx(2, "B2", LabelEnd),
          Tx(3, "C1", Cutoff.AddDays(-30)),
          Tx(4, "D1", Cutoff.AddDays(-30).AddSeconds(-1)),
      };

      var labels = LabelBuilder.Build(transactions, Cutoff, LabelEnd);

      // Single(조건)
      // 조건에 맞는 원소가 정확히 1개일 때 그 원소를 리턴함
      // 0개나 2개 이상이면 오류
      Assert.False(labels.Single(l => l.CustomerId == 1).Churn);
      Assert.True(labels.Single(l => l.CustomerId == 2).Churn);

      // 정확히 30일 전은 신규로 판정하고 1초라도 더 전이면 신규가 아닌 것으로 판정함
      Assert.True(labels.Single(l => l.CustomerId == 3).IsNew);
      Assert.False(labels.Single(l => l.CustomerId == 4).IsNew);
  }

  [Fact]
  public void Build_rejects_labelEnd_not_after_cutoff()
  {
      Assert.Throws<ArgumentException>(() => LabelBuilder.Build(new List<Transaction>(), Cutoff, Cutoff));
  }

  [DataFact]
  public void Build_matches_python_totals()
  {
      var labels = BuildFromData();

      Assert.Equal(4944, labels.Count);
      Assert.Equal(2372, labels.Count(l => l.Churn));
      Assert.Equal(105, labels.Count(l => l.IsNew));
      Assert.Equal(26033, labels.Sum(l => l.OrderCount));
  }

  // reference_features.csv 의 고객 번호, frequency, churn, is_new 와 고객별로 비교함
  [DataFact]
  public void Build_matches_reference_features_per_customer()
  {
      var labels = BuildFromData();
      var lines = File.ReadAllLines(TestData.ReferenceFeatures).ToList();
      var header = lines[0].Split(',');

      // Array.IndexOf 를 이용해 머리글에서 열 이름의 위치를 찾음
      int idIndex = Array.IndexOf(header, "customer_id");
      int frequencyIndex = Array.IndexOf(header, "frequency");
      int churnIndex = Array.IndexOf(header, "churn");
      int newIndex = Array.IndexOf(header, "is_new");

      Assert.Equal(labels.Count, lines.Count - 1);

      for (int i = 0 ; i < labels.Count ; i++)
    {
        var f = lines[i + 1].Split(',');

        Assert.Equal(int.Parse(f[idIndex]), labels[i].CustomerId);

        // frequency 는 파일에 소수로 표기되어 있어서 double 로 읽은 뒤 정수로 바꿈
        Assert.Equal((int)double.Parse(f[frequencyIndex], CultureInfo.InvariantCulture), labels[i].OrderCount);
        Assert.Equal(int.Parse(f[churnIndex]) == 1, labels[i].Churn);
        Assert.Equal(int.Parse(f[newIndex]) == 1, labels[i].IsNew);
    }
  }

  private static List<CustomerLabel> BuildFromData()
  {
      var transactions = CsvLoader.LoadTransactions(TestData.Transactions);
      return LabelBuilder.Build(transactions, Cutoff, LabelEnd);
  }
}