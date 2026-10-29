using Churn.Core.Models;

namespace Churn.Tests;

// 모델 클래스의 시험
// [Fact] 가 붙은 메서드 하나가 시험 1개임
public class ModelTests
{
    // 12개 * 6.95 = 83.4
    [Fact]
    public void Transaction_Revenue_is_quantity_times_price()
  {
      var t = new Transaction(13085, "489434", "85048", new DateTime(2009, 12, 1, 7, 45, 0), 12, 6.95, "United Kingdom");

      Assert.Equal(83.4, t.Revenue, 9);
  }

  // record struct 는 값이 모두 같으면 같은 거래로 봄
  [Fact]
  public void Transaction_with_same_values_are_equal()
  {
      var a = new Transaction(1, "489434", "85048", new DateTime(2009, 12, 1), 12, 6.95, "United Kingdom");
      var b = new Transaction(1, "489434", "85048", new DateTime(2009, 12, 1), 12, 6.95, "United Kingdom");

      Assert.Equal(a, b);
  }

  // enum 과 이름 목록 중 한쪽만 고치면 실패함
  [Fact]
  public void FeatureNames_count_matches_Feature_enum()
  {
      Assert.Equal(Enum.GetValues<Feature>().Length, FeatureNames.All.Count);
      Assert.Equal(20, FeatureNames.All.Count);
  }

  // 처음, 중간, 끝의 이름이 Feature 번호와 같은 칸에 있는지 확인함
  [Fact]
  public void FeatureNames_are_in_the_same_order_as_Feature_enum()
  {
      Assert.Equal("recency", FeatureNames.All[(int)Feature.Recency]);
      Assert.Equal("std_order_value", FeatureNames.All[(int)Feature.StdOrderValue]);
      Assert.Equal("orders_90d", FeatureNames.All[(int)Feature.Orders90d]);
      Assert.Equal("is_uk", FeatureNames.All[(int)Feature.IsUk]);
  }

  // 값을 넣은 칸의 값이 이름으로 읽히고, 넣지 않은 칸은 0 이어야 함
  [Fact]
  public void CustomerFeatures_indexer_returns_value_at_feature_position()
  {
      var values = new double[20];
      values[(int)Feature.Frequency] = 7;
      values[(int)Feature.IsUk] = 1;

      var c = new CustomerFeatures(13085, values, churn: true, isNew: false);

      Assert.Equal(13085, c.CustomerId);
      Assert.Equal(7, c[Feature.Frequency]);
      Assert.Equal(1, c[Feature.IsUk]);
      Assert.Equal(0, c[Feature.Recency]);
      Assert.Equal(20, c.Values.Length);
  }

  // 19개로 만들면 오류가 나야 함
  [Fact]
  public void CustomerFeatures_rejects_wrong_number_of_values()
  {
      Assert.Throws<ArgumentException>(() => new CustomerFeatures(1, new double[19], false, false));
  }

  // 만든 뒤에 바깥 배열을 고쳐도 고객의 값은 그대로여야 함
  [Fact]
  public void CustomerFeatures_keeps_its_own_copy_of_values()
  {
      var values = new double[20];
      var c = new CustomerFeatures(1, values, false, false);

      values[(int)Feature.Recency] = 99;

      Assert.Equal(0, c[Feature.Recency]);
  }
}
