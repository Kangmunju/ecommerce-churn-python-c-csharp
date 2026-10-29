using Churn.Core.Data;
using Churn.Core.Models;


namespace Churn.Tests;


// CsvLoader 의 시험
// 기대값은 tools/expected_values.py 의 출력
// C 시험과 동일한 값을 나타냄
public class CsvLoaderTests
{
    // [DataFact]: 데이터 파일이 있을 때만 실행함, void 는 돌려주는 값이 없다는 뜻
    [DataFact]
    public void LoadTransactions_matches_python_totals()
  {
      var rows = CsvLoader.LoadTransactions(TestData.Transactions);

      // Assert.Equal(기대값, 실제값): 다르면 실패, 숫자 뒤 L 은 long 표시
      // rows.Sum(t => 식): 각 원소 t 의 식 값을 더함 (t => ... 는 이름 없는 짧은 함수)
      // (long): 더하다가 int 범위를 넘지 않게 바꿈
      Assert.Equal(776577, rows.Count);
      Assert.Equal(11902294299L, rows.Sum(t => (long)t.CustomerId));
      Assert.Equal(10498435L, rows.Sum(t => (long)t.Quantity));
      Assert.Equal(1004902583108760L, rows.Sum(t => TestData.Seconds(t.InvoiceDate)));
      Assert.Equal(1706803087L, (long)Math.Round(rows.Sum(t => t.Revenue) * 100));
  }

  [DataFact]
  public void LoadTransactions_first_and_last_rows()
  {
      var rows = CsvLoader.LoadTransactions(TestData.Transactions);

      // rows[^1]: 맨 뒤 원소 (rows[rows.Count - 1] 과 같음)
      Assert.Equal(
          new Transaction(13085, "489434", "85048", new DateTime(2009, 12, 1, 7, 45, 0), 12, 6.95, "United Kingdom"),
          rows[0]);
      Assert.Equal(
          new Transaction(12680, "581587", "22138", new DateTime(2011, 12, 9, 12, 50, 0), 3, 4.95, "France"),
          rows[^1]);
  }

  [DataFact]
  public void LoadReturns_matches_python_totals()
  {
      var rows = CsvLoader.LoadReturns(TestData.Returns);

      Assert.Equal(18390, rows.Count);
      Assert.Equal(275994751L, rows.Sum(r => (long)r.CustomerId));
      Assert.Equal(23761154531760L, rows.Sum(r => TestData.Seconds(r.InvoiceDate)));
      // rows.Count(r => 조건): 조건이 참인 원소의 개수
      Assert.Equal(13309, rows.Count(r => r.InvoiceDate < new DateTime(2011, 6, 10)));
  }

  [DataFact]
  public void LoadReturns_first_and_last_rows()
  {
      var rows = CsvLoader.LoadReturns(TestData.Returns);

        Assert.Equal(new ReturnSlip(16321, "C489449", new DateTime(2009, 12, 1, 10, 33, 0)), rows[0]);
        Assert.Equal(new ReturnSlip(17315, "C581569", new DateTime(2011, 12, 9, 11, 58, 0)), rows[^1]);
  }


  // 아래는 임시 파일로 만든 잘못된 입력을 거절하는 지 확인하기 위해 작성함
  // 데이터 파일 없이도 실행됨

  [Fact]
  public void LoadReturns_reads_a_small_file()
  {
      var path = WriteTempFile("customer_id,invoice,invoice_date", "1,C100,2010-01-02 03:04:05");

      var rows = CsvLoader.LoadReturns(path);

      Assert.Single(rows);
      Assert.Equal(new ReturnSlip(1, "C100", new DateTime(2010, 1, 2, 3, 4, 5)), rows[0]);
  }

  [Fact]
  public void LoadReturns_rejects_wrong_header()
  {
      var path = WriteTempFile("id,invoice,date", "1,C100,2010-01-02 03:04:05");

      // Assert.Throws<오류 종류>(() => 식): 식을 실행했을 때 그 오류가 나야 통과함
      Assert.Throws<FormatException>(() => CsvLoader.LoadReturns(path));
  }

  [Fact]
  public void LoadReturns_rejects_wrong_column_count_and_says_line_number()
  {
      var path = WriteTempFile("customer_id,invoice,invoice_date", "1,C100,2010-01-02 03:04:05", "2,C101");

      // Throws 가 잡은 오류를 e 에 담아 메시지를 확인함
      var e = Assert.Throws<FormatException>(() => CsvLoader.LoadReturns(path));

      Assert.Contains("3행", e.Message);
  }

  [Fact]
  public void LoadReturns_rejects_bad_date_and_says_line_number()
  {
      var path = WriteTempFile("customer_id,invoice,invoice_date", "1,C100,2010/01/02");

      var e = Assert.Throws<FormatException>(() => CsvLoader.LoadReturns(path));

      Assert.Contains("2행", e.Message);
  }

  // params: 값을 쉼표로 여러 개 넘기면 배열로 묶어 받음 ("a", "b", "c")
  // Guid.NewGuid(): 겹치지 않는 임시 이름을 만듦
  private static string WriteTempFile(params string[] lines)
  {
      var path = Path.Combine(Path.GetTempPath(), $"churn_test_{Guid.NewGuid():N}.csv");
      File.WriteAllLines(path, lines);
      return path;
  }
}