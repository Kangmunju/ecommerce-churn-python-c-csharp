using System.Globalization;


namespace Churn.App;


// 비교 표의 한 줄
// Python: 정답지 값
// C: C 구현 값 (실행해 둔 값이 없으면 null)
// CSharp: 이번에 C# 이 계산한 값
// Tolerance: 파이썬과 특정 범위 이내로 다르면 일치로 판단할 것
// Format: 숫자를 찍는 모양을 설정함
public sealed record ComparisonRow(
    string Name,
    double Python,
    double? C,
    double CSharp,
    double Tolerance,
    string Format
)
{
    // 파이썬과의 차이가 허용 오차 이내인지 체크
    // 소수 계산 오차를 감안해 아주 작은 값(1e-12)을 더해서 비교함
    public bool Matches
    {
        get { return Math.Abs(CSharp - Python) <= Tolerance + 1e-12; }
    }

    // 표의 한 줄로 찍을 글자
    public string ToLine()
    {
        string python = Python.ToString(Format, CultureInfo.InvariantCulture);
        string csharp = CSharp.ToString(Format, CultureInfo.InvariantCulture);
        string c = "-";
        if (C.HasValue)
        {
            c = C.Value.ToString(Format, CultureInfo.InvariantCulture);
        }
        string tolerance = Tolerance.ToString("G", CultureInfo.InvariantCulture);
        string result = "벗어남";
        if (Matches)
        {
            result = "일치";
        }
        return $"{Name,-22} {python,12} {c,12} {csharp,12}   ±{tolerance,-8} {result}";
    }
}
