using System.Globalization;
using Churn.Core.Data;
using Churn.Core.Features;
using Churn.Core.Labels;
using Churn.Core.Models;


namespace Churn.Tests;


// FeatureBuilder 의 시험
// 작은 데이터는 손으로 계산한 값
// 실제 데이터는 reference_features.csv 와 비교함
public class FeatureBuilderTests
{
    private static readonly DateTime Cutoff = new(2011, 6, 10);
    private static readonly DateTime LabelEnd = new(2011, 12, 10);

    // 시험용 거래 한 줄
    private static Transaction Line(int customer, string invoice, string stock, DateTime time, int quantity, double price, string country = "United Kingdom")
  {
      return new Transaction(customer, invoice, stock, time, quantity, price, country);
  }

  private static ReturnSlip Slip(int customer, string invoice, DateTime time)
  {
      return new ReturnSlip(customer, invoice, time);
  }

  private static List<CustomerFeatures> Build(List<Transaction> transactions, List<ReturnSlip> returns)
  {
      var labels = LabelBuilder.Build(transactions, Cutoff, LabelEnd);
      return FeatureBuilder.Build(transactions, returns, labels, Cutoff);
  }

  // 고객 1: 주문 3건, 마지막 줄의 국가는 France
  // A1 2011-03-01 10:00 25(10*2 + 5*1), 수량 3
  // A2 2011-04-01 10:00  20, 수량 4
  // A3 2011-05-20 10:00  55, 수량 1, France
  // 주문 간격: A1 -> A2 는 31일, A2 -> A3 는 49일
  // 반품 송장 C1: 두 줄이지만 같은 송장이라 1건으로 셈
  // 반품 송장 C2: 기준일(2011-06-10) 이후(2011-07-01)라 제외함
  private static (List<Transaction>, List<ReturnSlip>) CustomerOneData()
  {
      var transactions = new List<Transaction>
      {
          Line(1, "A1", "S1", new DateTime(2011, 3, 1, 10, 0, 0), 2, 10.0),
          Line(1, "A1", "S2", new DateTime(2011, 3, 1, 10, 0, 0), 1, 5.0),
          Line(1, "A2", "S1", new DateTime(2011, 4, 1, 10, 0, 0), 4, 5.0),
          Line(1, "A3", "S3", new DateTime(2011, 5, 20, 10, 0, 0), 1, 55.0, "France"),
      };
      var returns = new List<ReturnSlip>
      {
          Slip(1, "C1", new DateTime(2011, 4, 2)),
          Slip(1, "C1", new DateTime(2011, 4, 3)),
          Slip(1, "C2", new DateTime(2011, 7, 1)),
      };
      return (transactions, returns);

  }

  [Fact]
  public void Build_computes_all_features_for_a_multi_order_customer()
  {
      var (transactions, returns) = CustomerOneData();

      var c = Build(transactions, returns).Single();

      double mean = 100.0 / 3.0;
      double std = Math.Sqrt((Math.Pow(25 - mean, 2) + Math.Pow(20 - mean, 2) + Math.Pow(55 - mean, 2)) / 2.0);

      Assert.Equal(20, c[Feature.Recency]);
      Assert.Equal(3, c[Feature.Frequency]);
      Assert.Equal(100.0, c[Feature.Monetary], 9);
      Assert.Equal(mean, c[Feature.AvgOrderValue], 9);
      Assert.Equal(std, c[Feature.StdOrderValue], 9);
      Assert.Equal(55.0, c[Feature.MaxOrderValue], 9);
      Assert.Equal(8, c[Feature.TotalItems]);
      Assert.Equal(100, c[Feature.Tenure]);
      Assert.Equal(80, c[Feature.ActiveSpan]);
      Assert.Equal(40.0, c[Feature.GapMean], 9);
      Assert.Equal(Math.Sqrt(162.0), c[Feature.GapStd], 9);
      Assert.Equal(49, c[Feature.GapMax]);
      Assert.Equal(0.5, c[Feature.RecencyRatio], 9);
      Assert.Equal(2, c[Feature.Orders90d]);
      Assert.Equal(75.0, c[Feature.Revenue90d], 9);
      Assert.Equal(0.75, c[Feature.Share90d], 9);
      Assert.Equal(3, c[Feature.NProducts]);
      Assert.Equal(1, c[Feature.NReturns]);
      Assert.Equal(1.0 / 3.0, c[Feature.ReturnRate], 9);
      Assert.Equal(0, c[Feature.IsUk]);
  }

  
  // 주문이 1번인 고객
  // 표준편차와 간격 표준편차 = 0
  // 간격 평균과 최댓값 = tenure
  [Fact]
  public void Build_handles_a_single_order_customer()
  {
      var transactions = new List<Transaction>
      {
          Line(2, "B1", "S1", new DateTime(2011, 6, 1, 12, 0, 0), 1, 10.0),
      };

      var c = Build(transactions, new List<ReturnSlip>()).Single();

      Assert.Equal(8, c[Feature.Recency]);
      Assert.Equal(8, c[Feature.Tenure]);
      Assert.Equal(0, c[Feature.ActiveSpan]);
      Assert.Equal(0.0, c[Feature.StdOrderValue]);
      Assert.Equal(8, c[Feature.GapMean]);
      Assert.Equal(0, c[Feature.GapStd]);
      Assert.Equal(8, c[Feature.GapMax]);
      Assert.Equal(1.0, c[Feature.RecencyRatio], 9);
      Assert.Equal(1.0, c[Feature.Share90d], 9);
      Assert.Equal(0, c[Feature.ReturnRate]);
      Assert.Equal(1, c[Feature.IsUk]);
  }

  // 같은 날 주문 2건: 간격 평균이 0이면 recency_ratio 를 1.0으로 체크함
  // 반품이 주문보다 많으면 return_rate 는 1.0에서 자름
  [Fact]
  public void Build_handles_zero_gap_and_return_rate_cap()
  {
      var transactions = new List<Transaction>
      {
          Line(3, "C1", "S1", new DateTime(2011, 1, 1, 9, 0, 0), 1, 10.0),
          Line(3, "C2", "S1", new DateTime(2011, 1, 1, 15, 0, 0), 1, 10.0)
      };
      var returns = new List<ReturnSlip>
      {
          Slip(3, "R1", new DateTime(2011, 2, 1)),
          Slip(3, "R2", new DateTime(2011, 2, 2)),
          Slip(3, "R3", new DateTime(2011, 2, 3))
      };

      var c = Build(transactions, returns).Single();

      Assert.Equal(0, c[Feature.GapMean]);
      Assert.Equal(1.0, c[Feature.RecencyRatio], 9);
      Assert.Equal(3, c[Feature.NReturns]);
      Assert.Equal(1.0, c[Feature.ReturnRate], 9);
  }

  // 간격이 1일인데 오래 쉬고 있으면 recency_ratio 는 50에서 자르도록 함
  [Fact]
  public void Build_caps_recency_ratio_at_50()
  {
      var transactions = new List<Transaction>
      {
          Line(4, "D1", "S1", new DateTime(2010, 1, 1), 1, 10.0),
          Line(4, "D2", "S1", new DateTime(2010, 1, 2), 1, 10.0)
      };

      var c = Build(transactions, new List<ReturnSlip>()).Single();

      Assert.True(c[Feature.Recency] > 50);
      Assert.Equal(50.0, c[Feature.RecencyRatio], 9);
  }

  // 기준일 이후의 거래와 반품을 더해도 피처는 그대로여야 함
  // 미래를 보지 않도록 주의하며 작성함
  [Fact]
  public void Build_ignores_everything_after_cutoff()
  {
      var (transactions, returns) = CustomerOneData();
      var before = Build(transactions, returns).Single();

      var withFuture = new List<Transaction>(transactions)
      {
          Line(1, "A9", "S9", new DateTime(2011, 7, 1), 100, 100.0),
          Line(1, "A9", "S9", Cutoff, 1, 1.0),
      };
      var withFutureReturns = new List<ReturnSlip>(returns)
      {
          Slip(1, "C9", Cutoff)
      };
      var after = Build(withFuture, withFutureReturns).Single();

      Assert.Equal(before.Values.ToArray(), after.Values.ToArray());
  }

  [DataFact]
  public void Build_matches_reference_features()
  {
      var transactions = CsvLoader.LoadTransactions(TestData.Transactions);
      var returns = CsvLoader.LoadReturns(TestData.Returns);
      var labels = LabelBuilder.Build(transactions, Cutoff, LabelEnd);

      var features = FeatureBuilder.Build(transactions, returns, labels, Cutoff);

      var lines = File.ReadLines(TestData.ReferenceFeatures).ToList();
      var header = lines[0].Split(',');
      int idIndex = Array.IndexOf(header, "customer_id");
      int churnIndex = Array.IndexOf(header, "churn");
      int newIndex = Array.IndexOf(header, "is_new");

      // 피처 이름으로 열 위치를 찾음
      // 이름이 하나라도 없으면 -1을 리턴함
      var columns = FeatureNames.All.Select(name => Array.IndexOf(header, name)).ToArray();
      Assert.DoesNotContain(-1, columns);
      Assert.Equal(features.Count, lines.Count - 1);

      double maxError = 0.0;
      int worst = 0;

      for (int i = 0 ; i < features.Count ; i++)
    {
        var f = lines[i + 1].Split(',');

        Assert.Equal(int.Parse(f[idIndex]), features[i].CustomerId);
        Assert.Equal(int.Parse(f[churnIndex]) == 1, features[i].Churn);
        Assert.Equal(int.Parse(f[newIndex]) == 1, features[i].IsNew);

        for (int k = 0 ; k < columns.Length ; k++)
      {
          double expected = double.Parse(f[columns[k]], CultureInfo.InvariantCulture);
          double error = Math.Abs(features[i].Values[k] - expected) / Math.Max(1.0, Math.Abs(expected));
          if (error > maxError)
        {
            maxError = error;
            worst = k;
        }
      }
    }

    Assert.True(maxError < 1e-9, $"최대 상대 오차 {maxError:E2} ({FeatureNames.All[worst]})");
  }
}