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
