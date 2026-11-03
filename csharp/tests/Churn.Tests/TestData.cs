using Churn.Core.Data;
using Churn.Core.Features;
using Churn.Core.Labels;
using Churn.Core.Models;


namespace Churn.Tests;


// 시험이 쓰는 데이터 파일의 위치
// 시험 프로그램은 csharp/tests/Churn.Tests/bin/... 에서 실행됨
// 위쪽 폴더로 올라가며 csharp/Churn.sln 이 있는 폴더(저장소 맨 위)를 찾도록 함
public static class TestData
{
    private static readonly string Root = FindRoot();

    // Path.Combine: 폴더 이름들을 운영체제에 맞는 구분자로 이어 붙임
    public static readonly string ExportDir = Path.Combine(Root, "data", "processed", "export_for_c");

    public static readonly string Transactions = Path.Combine(ExportDir, "clean_transactions.csv");

    public static readonly string Returns = Path.Combine(ExportDir, "returns.csv");

    public static readonly string ReferenceFeatures = Path.Combine(ExportDir, "reference_features.csv");

    public static readonly string ReferenceSplit = Path.Combine(ExportDir, "reference_split.csv");

    public static readonly string ReferenceCoefficients = Path.Combine(ExportDir, "reference_coefficients.csv");

    public static readonly string ReferenceTestPred = Path.Combine(ExportDir, "reference_test_pred.csv");

    // 실제 데이터로 만든 피처를 파이썬과 같은 학습, 시험으로 나눈 것
    // Lazy: 처음 쓸 때 한 번만 만들고 이후 시험들이 같은 결과를 사용함
    // 만드는 데 1 ~ 2초 걸릴 것으로 예상됨
    private static readonly Lazy<(List<CustomerFeatures>, List<CustomerFeatures>)> SplitData = new(LoadSplit);

    // Item1: 학습 고객, Item2: 시험 고객 (튜플의 첫 번째, 두 번째 값)
    public static List<CustomerFeatures> Train => SplitData.Value.Item1;

    public static List<CustomerFeatures> Test => SplitData.Value.Item2;

    // 거래, 반품 CSV 읽기 -> 라벨 -> 피처 -> reference_split.csv 로 학습, 시험 나누기
    private static (List<CustomerFeatures>, List<CustomerFeatures>) LoadSplit()
  {
      var cutoff = new DateTime(2011, 6, 10);
      var transactions = CsvLoader.LoadTransactions(Transactions);
      var returns = CsvLoader.LoadReturns(Returns);
      var labels = LabelBuilder.Build(transactions, cutoff, new DateTime(2011, 12, 10));
      var features = FeatureBuilder.Build(transactions, returns, labels, cutoff);

      // reference_split.csv: customer_id,split (split 은 train 또는 test)
      var testIds = new HashSet<int>();
      foreach (var line in File.ReadLines(ReferenceSplit).Skip(1))
    {
        var f = line.Split(',');
        if (f[1] == "test")
      {
          testIds.Add(int.Parse(f[0]));
      }
    }

    // 시험 고객 번호 목록에 있으면 시험, 없으면 학습으로 나눔
    // 파이썬은 무작위로 나눴고 C# 은 그 결과를 파일에서 읽어 그대로 씀
    var train = new List<CustomerFeatures>();
    var test = new List<CustomerFeatures>();
    foreach (var c in features)
    {
        if (testIds.Contains(c.CustomerId))
      {
          test.Add(c);
      }
      else
      {
          train.Add(c);
      }
    }
    return (train, test);
  }

    private static string FindRoot()
  {
      // AppContext.BaseDirectory: 실행 중인 시험 프로그램이 있는 폴더
      var dir = new DirectoryInfo(AppContext.BaseDirectory);
      while (dir != null)
    {
        if (File.Exists(Path.Combine(dir.FullName, "csharp", "Churn.sln")))
      {
          return dir.FullName;
      }
      dir = dir.Parent;
    }
    throw new DirectoryNotFoundException("저장소 맨 위 폴더(csharp/Churn.sln 이 있는 곳)를 찾지 못함");
  }


  // 1970-01-01 00:00:00 부터 지난 초
  // C 시험의 '초 합계' 와 같은 기준을 적용함
  // DateTime 끼리 빼면 시간 간격이 되고 TotalSeconds 가 그 간격을 초로 보여 줌
  public static long Seconds(DateTime time)
  {
      return (long)(time - new DateTime(1970, 1, 1)).TotalSeconds;
  }
}


// 데이터 파일이 있을 때만 실행하는 시험
// data/ 는 git에 올리지 않음
// 새로 내려받은 저장소에서 찾을 수 없음에 주의
// python tools/export_for_c.py로 만들었음
// 이름이 Attribute 로 끝나면 [DataFact] 처럼 Attribute 를 빼고 시험 위에 붙일 수 있음
public sealed class DataFactAttribute: FactAttribute
{
    public DataFactAttribute()
  {
      if (!File.Exists(TestData.Transactions) || !File.Exists(TestData.Returns))
    {
        // Skip 에 이유를 적으면 실패 대신 "건너뜀" 으로 표시됨
        Skip = "데이터 파일 없음 (python tools/export_for_c.py 로 생성)";
    }
  }
}
