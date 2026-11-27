using System.Globalization;
using System.Text;
using Churn.App;
using Churn.Core.Pipeline;


// 사용법
// dotnet run --project csharp/src/Churn.App -c Release [데이터 폴더]
// 데이터 폴더를 적지 않으면 저장소 맨 위에서 data/processed/export_for_c 를 찾음
Console.OutputEncoding = Encoding.UTF8;

string exportDir;
if (args.Length > 0)
{
    exportDir = args[0];
}
else
{
    exportDir = FindExportDir();
}
if (!Directory.Exists(exportDir))
{
    Console.WriteLine($"데이터 폴더가 없음: {exportDir}");
    Console.WriteLine("python tools/export_for_c.py 로 먼저 만들 것");
    return 2;
}

Console.WriteLine($"데이터 폴더: {exportDir}");
Console.WriteLine("파이프라인 실행 중 ...");
var r = PipelineRunner.Run(PipelineOptions.ForDirectory(exportDir));

// 단계별 시간
Console.WriteLine();
Console.WriteLine("[단계별 시간]");
double total = 0.0;
foreach (var t in r.Times)
{
    Console.WriteLine($"  {t.Name,-24} {t.Seconds,7:F3} 초");
    total += t.Seconds;
}
Console.WriteLine($"  {"합계",-24} {total,7:F3} 초");

// 기대이익 곡선
Console.WriteLine();
Console.WriteLine("[임계값별 기대이익 (시험 고객, 쿠폰 1장 맞게 +14.05 파운드 / 헛되이 -5.00)]");
Console.WriteLine("  임계값   발송   TP   FP    기대이익");
foreach (var p in r.Curve)
{
    Console.WriteLine($"  {p.Threshold,5:F2} {p.Sent,6} {p.Tp,4} {p.Fp,4} {p.Profit,10:F2}");
}

// 비교 표: 파이썬 정답지 / C 구현 / 이번 C# 결과
// 파이썬 값은 reports/*.csv, reference_summary.csv, C 값은 C 구현의 마지막 실행 결과를 옮겨 적은 것
var m = r.Logistic;
var rows = new List<ComparisonRow>
{
    new("고객 수", 4944, 4944, r.Customers, 0, "F0"),
    new("이탈 고객", 2372, 2372, r.Churned, 0, "F0"),
    new("신규 고객", 105, 105, r.NewCustomers, 0, "F0"),
    new("학습 / 시험 고객 합", 4944, 4944, r.TrainCount + r.TestCount, 0, "F0"),
    new("AUC", 0.781834, 0.781821, m.Auc, 0.001, "F6"),
    new("PR-AUC", 0.750204, null, m.PrAuc, 0.001, "F6"),
    new("정확도", 0.712938, null, m.Accuracy, 0.002, "F6"),
    new("정밀도", 0.684278, null, m.Precision, 0.002, "F6"),
    new("재현율", 0.745787, null, m.Recall, 0.002, "F6"),
    new("F1", 0.713710, null, m.F1, 0.002, "F6"),
    new("Brier", 0.190717, null, m.Brier, 0.001, "F6"),
    new("혼동행렬 TN", 527, 526, m.Matrix.Tn, 2, "F0"),
    new("혼동행렬 FP", 245, 246, m.Matrix.Fp, 2, "F0"),
    new("혼동행렬 FN", 181, 180, m.Matrix.Fn, 2, "F0"),
    new("혼동행렬 TP", 531, 532, m.Matrix.Tp, 2, "F0"),
    new("유지 고객 매출 중앙값", 635.01, 635.01, r.RetainedMedian, 0.01, "F2"),
    new("최적 임계값", 0.30, 0.30, r.Best.Threshold, 0.001, "F2"),
    new("최적 기대이익", 6969.34, 6969.34, r.Best.Profit, 0.01, "F2"),
    new("임계값 0.50 기대이익", 6235.71, 6244.76, r.AtHalf.Profit, 20, "F2"),
    new("OOF 임계값", 0.25, null, r.OofThreshold, 0.001, "F2"),
    new("OOF 임계값 기대이익", 6908.90, null, r.OofApplied.Profit, 30, "F2"),
    new("로지스틱 CV 평균 AUC", 0.8044, null, r.LogisticCv.MeanAuc, 0.003, "F4"),
    new("MLP CV 평균 AUC", 0.8046, null, r.MlpCv.MeanAuc, 0.005, "F4"),
    new("MLP 시드 평균 AUC", 0.7836, null, r.MlpSeedAucs.Average(), 0.003, "F4"),
};

Console.WriteLine();
Console.WriteLine("[파이썬 / C / C# 비교]");
Console.WriteLine($"{"항목",-22} {"파이썬",12} {"C",12} {"C#",12}   {"허용 오차",-9} 결과");
int mismatches = 0;
foreach (var row in rows)
{
    Console.WriteLine(row.ToLine());
    if (!row.Matches)
    {
        mismatches++;
    }
}

Console.WriteLine();
if (mismatches == 0)
{
    Console.WriteLine($"모든 {rows.Count}개 항목이 파이썬과 허용 오차 안에서 일치함");
    return 0;
}
Console.WriteLine($"{rows.Count}개 중 {mismatches}개가 허용 오차를 벗어남");
return 1;


// 실행 파일이 있는 폴더에서 위로 올라가며 csharp/Churn.sln 이 있는 폴더를 찾고, 그 아래 데이터 폴더를 돌려줌
static string FindExportDir()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir != null)
    {
        if (File.Exists(Path.Combine(dir.FullName, "csharp", "Churn.sln")))
        {
            return Path.Combine(dir.FullName, "data", "processed", "export_for_c");
        }
        dir = dir.Parent;
    }
    return Path.Combine("data", "processed", "export_for_c");
}
