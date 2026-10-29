using System.Globalization;
using Churn.Core.Models;


namespace Churn.Core.Data;


// CSV 파일을 읽어 모델 목록으로 바꿈
// 거래와 반품이 파일 읽기 코드를 나눠 쓰도록 제네릭(ReadRows<T>)으로 작성함
// C 에서는 타입마다 같은 코드를 복사했던 부분
// 데이터에 쉼표나 따옴표가 들어 있지 않아 쉼표로만 나눔
public static class CsvLoader
{
    // const: 바뀌지 않는 값, MM 은 월 mm 은 분
    private const string DateFormat = "yyyy-MM-dd HH:mm:ss";

    private static readonly string[] TransactionsHeader =
      ["customer_id", "invoice", "stock_code", "invoice_date", "quantity", "price", "country"];

    private static readonly string[] ReturnHeader =
      ["customer_id", "invoice", "invoice_date"];

    // 메서드 모양: 접근범위 종류 반환형 이름(매개변수)
    // static 이라 객체 없이 CsvLoader.LoadTransactions(...) 로 부름
    public static List<Transaction> LoadTransactions(string path)
  {
      return ReadRows(path, TransactionsHeader, ParseTransaction);
  }

  public static List<ReturnSlip> LoadReturns(string path)
  {
      return ReadRows(path, ReturnHeader, ParseReturn);
  }

  // 한 줄(fields)을 거래 한 건으로 바꿈
  // 이름: 값 은 이름 붙은 인자, InvariantCulture 는 나라 설정과 무관하게 같은 규칙으로 읽음
  private static Transaction ParseTransaction(string[] fields)
  {
      return new Transaction(
          CustomerId: int.Parse(fields[0], CultureInfo.InvariantCulture),
          Invoice: fields[1],
          StockCode: fields[2],
          InvoiceDate: ParseDate(fields[3]),
          Quantity: int.Parse(fields[4], CultureInfo.InvariantCulture),
          Price: double.Parse(fields[5], CultureInfo.InvariantCulture),
          Country: fields[6]
      );
  }

  private static ReturnSlip ParseReturn(string[] fields)
  {
      return new ReturnSlip(
          CustomerId: int.Parse(fields[0], CultureInfo.InvariantCulture),
          Invoice: fields[1],
          InvoiceDate: ParseDate(fields[2])
      );
  }


  // 'YYYY-MM-DD HH:MM:SS' 글자를 날짜로 바꿈
  // 형식이 다르면 FormatException 이 나도록 작성함
  private static DateTime ParseDate(string text)
  {
      return DateTime.ParseExact(text, DateFormat, CultureInfo.InvariantCulture);
  }


  // 파일의 머리글을 확인하고 나머지 줄을 parse 로 바꿔 목록으로 돌려줌
  // 오류가 나면 몇 번째 줄인지 알려 줌
  // <T>: 부를 때 정해지는 타입 (거래면 Transaction, 반품이면 ReturnSlip)
  // Func<string[], T>: 글자 배열을 받아 T 를 돌려주는 함수, 함수 이름만 넘김
  private static List<T> ReadRows<T>(string path, string[] header, Func<string[], T> parse)
  {
      var rows = new List<T>();
      int lineNumber = 0;

      // ReadLines: 한 줄씩 읽음 (ReadAllLines 는 전체를 한 번에 메모리에 올림)
      foreach (var line in File.ReadLines(path))
    {
        lineNumber++;

        // continue: 이번 줄은 건너뜀
        if (line.Length == 0)
      {
          continue;
      }

      // Split(','): 쉼표 기준으로 나눠 배열로 만듦
      var fields = line.Split(',');

      if (lineNumber == 1)
      {
          // SequenceEqual: 배열 내용이 순서까지 같은지 비교함
          if (!fields.SequenceEqual(header))
        {
            throw new FormatException(
                $"{path} 머리글이 다름 (기대: {string.Join(",", header)}, 실제: {line})"
            );
        }
        continue;
      }

      if (fields.Length != header.Length)
      {
          throw new FormatException(
              $"{path} {lineNumber}행: 열이 {header.Length}개여야 함 (받은 개수: {fields.Length})"
          );
      }

      // try 안에서 오류가 나면 catch 가 받아 줄 번호를 붙여 다시 던짐
      try
      {
          rows.Add(parse(fields));
      }
      catch (FormatException e)
      {
          throw new FormatException($"{path} {lineNumber}행: {e.Message}", e);
      }
    }

    return rows;
  }
}