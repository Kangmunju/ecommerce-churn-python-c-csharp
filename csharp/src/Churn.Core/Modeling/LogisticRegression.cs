using Churn.Core.Models;


namespace Churn.Core.Modeling;


// L2 규제 로지스틱 회귀
// 뉴턴법
// 찾는 값: L(w) = 1/2 * (계수 제곱합) + C * (고객별 로그 손실의 합) 을 가장 작게 만드는 w
// 절편에는 규제를 붙이지 않음
// sklearn LogisticRegression 과 같음
// C 에서는 가우스 소거법과 배열 크기를 직접 관리했던 부분을 double[,] 와 IClassifier 로 정리함
public sealed class LogisticRegression : IClassifier
{

    // 뉴턴법 반복 횟수의 상한
    // 일반적으로 10회 안팎에 끝남
    private const int MaxIterations = 100;

    // 한 번의 반복에서 w 가 가장 크게 움직인 거리가 이 값보다 작으면 멈추도록 함
    private const double StepTolerance = 1e-10;

    // 규제 세기의 역수
    // 작을수록 계수를 더 억누르는 것
    private readonly double _c;

    // 학습 고객의 평균과 표준편차를 기억함
    // 예측할 때도 같은 기준으로 표준화하기 위함
    private readonly StandardScaler _scaler = new();

    // 학습이 끝난 계수 20개
    // Fit 전에는 빈 배열이므로 길이로 학습 여부를 알 수 있음
    private double[] _coefficients = Array.Empty<double>();

    public LogisticRegression(double c = 1.0)
  {
      // 아래 Fit 에서 1 / c 로 규제 세기를 구하므로 0 이하는 거부함
      if (c <= 0.0)
    {
         throw new ArgumentOutOfRangeException(nameof(c), "C 는 0보다 커야 함");
    }
    _c = c;
  }

  // 표준화된 피처에 대한 계수 20개로 Feature 번호 순서를 나타냄
  public IReadOnlyList<double> Coefficients => _coefficients;

  public double Intercept { get; private set; }

  // 수렴까지 걸린 뉴턴법 반복 횟수를 구함
  public int Iterations { get; private set; }

  // 시험에서 학습 때 쓴 평균과 표준편차를 확인하기 위해 공개함
  public StandardScaler Scaler => _scaler;

  // 학습 순서
  // 1. 학습 고객의 평균과 표준편차로 피처를 표준화함
  // 2. 계수 20개와 절편을 모두 0에서 시작함
  // 3. 지금의 계수로 고객마다 이탈 확률 p 를 구함
  // 4. p 와 정답의 차이로 기울기(grad)와 곡선의 휘어짐(hess)을 계산함
  // 5. H * step = grad 를 풀어 계수를 step 만큼 이동함
  // 6. 3 ~ 5 를 되풀이하다가 이동 거리가 아주 작아지면 멈춤
  public void Fit(IReadOnlyList<CustomerFeatures> train)
  {
      if (train.Count == 0)
    {
        throw new ArgumentException("학습할 고객이 없음", nameof(train));
    }

    // 학습 고객만으로 평균과 표준편차를 구함
    _scaler.Fit(train);

    // n: 학습 고객 수
    // features: 피처 개수 (20)
    // size: 찾을 값의 개수, 계수 20개 + 절편 1개 = 21
    int n = train.Count;
    int features = FeatureNames.All.Count;
    int size = features + 1;

    // 표준화한 피처 20개 뒤에 값이 1인 칸(절편용)을 붙여 고객마다 한 줄로 만들어 둠
    // 반복마다 다시 계산하지 않도록 한 번만 만듦
    // rows[i][k]: i번째 고객의 k번째 값, k가 20이면 항상 1.0 (절편용)
    // answers[i]: i번째 고객의 정답 y, 이탈이면 1 유지면 0
    var rows = new double[n][];
    var answers = new double[n];
    for (int i = 0 ; i < n ; i++)
    {
        var row = new double[size];
        Array.Copy(_scaler.Transform(train[i]), row, features);
        row[features] = 1.0;
        rows[i] = row;

        // 배열의 기본값이 0이므로 이탈 고객만 1로 바꿈
        if (train[i].Churn)
      {
          answers[i] = 1.0;
      }
    }

    // w[0] ~ w[19] 는 계수
    // w[20] 은 절편
    // 모두 0에서 시작함
    // 모든 고객의 확률이 0.5
    var w = new double[size];
    // reg: 규제 세기, C 가 1이면 1, C 가 0.01이면 100으로 계수를 훨씬 세게 억누름
    // converged: 이동 거리가 충분히 작아져서 멈췄는지 기록함
    double reg = 1.0 / _c;
    bool converged = false;

    for (int iter = 1 ; iter <= MaxIterations ; iter++)
    {

        // 반복할 때마다 새 w 로 grad 와 hess 를 처음부터 다시 계산함
        // grad: L 을 w 의 각 값으로 미분한 기울기 -> 최적점에서는 모두 0
        //       a번째 값을 조금 키우면 L 이 얼마나 커지는지를 나타냄
        // hess: 기울기가 w 에 따라 변하는 정도 (곡선의 휘어짐)
        //       21 x 21 표, 많이 휘어 있을수록 조심스럽게 움직여야 함
        var grad = new double[size];
        var hess = new double[size, size];

        for (int i = 0 ; i < n ; i++)
      {
          var row = rows[i];

          // score: 이 고객의 점수 = (계수 * 피처값) 을 모두 더한 값
          // 마지막 칸은 값이 1이라 절편이 그대로 더해짐
          double score = 0.0;
          for (int a = 0 ; a < size ; a++)
        {
            score += w[a] * row[a];
        }

        // p: 점수를 0 ~ 1 확률로 바꾼 값, 점수 0이면 0.5, 점수 2이면 약 0.88, 점수 -2이면 약 0.12
        // err: 확률과 정답의 차이 (p - y)
        //      이탈(y=1)인데 p=0.2 이면 -0.8, 유지(y=0)인데 p=0.2 이면 +0.2
        //      많이 틀릴수록 절댓값이 크고 부호가 고칠 방향을 알려줌
        // weight: 판단이 애매할수록 커지는 p(1-p)
        //         p=0.5 이면 0.25 로 최대, p=0.9 이면 0.09
        //         이미 확신하는 고객은 계수를 움직이는 힘이 작음
        double p = Sigmoid(score);
        double err = p - answers[i];
        double weight = p * (1.0 - p);

        for (int a = 0 ; a < size ; a++)
        {
            // a번째 기울기 += 틀린 정도(err) * 이 고객의 a번째 값
            // 고객 전체를 더하면 a번째 값을 키웠을 때 전체 손실이 변하는 정도가 됨
            grad[a] += err * row[a];

            // (a, b) 와 (b, a) 는 동일함
            // b <= a 인 아래 삼각형만 계산하면 됨
            for (int b = 0 ; b <= a ; b++)
          {
              // (a, b) 칸 += weight * a번째 값 * b번째 값
              hess[a, b] += weight * row[a] * row[b];
          }
        }
      }

      // 아래 삼각형만 채웠으므로 위 삼각형에 그대로 복사함
      for (int a = 0 ; a < size ; a++)
      {
          for (int b = a+1 ; b < size ; b++)
        {
            hess[a, b] = hess[b, a];
        }
      }

      // 규제 항: L 의 1/2 * reg * w^2 을 미분하면 기울기에 reg * w, 헤세 행렬 대각선에 reg
      // 계수에만 붙이고 마지막 칸인 절편에는 붙이지 않음
      for (int k = 0 ; k < features ; k++)
      {
          grad[k] += reg * w[k];
          hess[k, k] += reg;
      }

      // 뉴턴법: 기울기와 휘어짐을 함께 보고 한 번에 크게 이동함
      // 값이 하나뿐이라면 step = grad / hess, 예를 들어 기울기 6, 휘어짐 3 이면 2만큼 이동함
      // 값이 21개이므로 나눗셈 대신 H * step = grad 라는 연립방정식을 풀어야 함
      // Solve 가 끝나면 grad 자리에 step 이 들어 있음
      if (!Solve(hess, grad))
      {
          throw new InvalidOperationException($"헤세 행렬을 풀 수 없음 (반복 {iter})");
      }

      // w 에서 step 을 빼서 이동함
      // maxStep: 이번에 가장 크게 움직인 거리
      // 거의 안 움직였다면 이미 최적점에 도착한 것
      double maxStep = 0.0;
      for (int a = 0 ; a < size ; a++)
      {
          w[a] -= grad[a];
          if (Math.Abs(grad[a]) > maxStep)
        {
            maxStep = Math.Abs(grad[a]);
        }
      }

      Iterations = iter;
      if (maxStep < StepTolerance)
      {
          converged = true;
          break;
      }
    }

    if (!converged)
    {
        throw new InvalidOperationException($"{MaxIterations}회 안에 수렴하지 않음");
    }

    // 수렴했을 때만 결과를 저장함
    // w 의 앞 20개는 계수, 마지막 1개는 절편
    var coefficients = new double[features];
    Array.Copy(w, coefficients, features);
    _coefficients = coefficients;
    Intercept = w[features];
  }

  // 고객 한 명의 이탈 확률을 구함
  public double PredictProbability(CustomerFeatures customer)
  {
      // Fit 전에는 계수가 빈 배열이므로 예측할 수 없음
      if (_coefficients.Length == 0)
    {
        throw new InvalidOperationException("Fit 을 먼저 호출해야 함");
    }

    // 표준화할 때 시험 고객의 평균이 아니라 학습 때 저장해 둔 평균과 표준편차를 씀
    // 시험 고객의 통계를 쓰면 누수가 생기기 때문
    var z = _scaler.Transform(customer);

    // 점수 = 절편 + (계수 * 표준화한 피처값) 의 합
    // 이 점수를 시그모이드로 확률로 바꿈
    double score = Intercept;
    for (int k = 0 ; k < z.Length ; k++)
    {
        score += _coefficients[k] * z[k];
    }
    return Sigmoid(score);
  }

  // 시그모이드: 어떤 실수든 0 과 1 사이 확률로 바꿈
  // 1 / (1 + e^-z), 예: z=0 이면 0.5, z=2 이면 약 0.88, z=-2 이면 약 0.12
  // z 가 크면 1에 가깝고 작으면 0에 가까움
  // z 가 -1000 처럼 큰 음수이면 Math.Exp(-z) 가 e^1000 이 되어 무한대로 넘침
  // 그래서 음수일 때는 e^z / (1 + e^z) 로 계산함
  // 수학적으로 같은 식이고, e^z 는 0에 가까운 작은 수라 안전함
  private static double Sigmoid(double z)
  {
      if (z >= 0.0)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }

    double e = Math.Exp(z);
    return e / (1.0 + e);
  }

  // 연립방정식 A x = b 를 풀어 x 를 b 자리에 덮어씀 (가우스 소거법)
  // 예: 2x + y = 5, x + 3y = 10 이면
  //     첫 식의 절반을 둘째 식에서 빼서 x 를 없애면 2.5y = 7.5, 그러므로 y = 3
  //     y 를 첫 식에 넣으면 x = 1
  // 위쪽 식으로 아래 식의 앞쪽 미지수를 차례로 지워 삼각형 모양을 만들고(소거)
  // 맨 아래 식부터 대입하며 올라감(역대입)
  // 행렬 계산 라이브러리가 없어 직접 만들었고, 역행렬을 구하는 것보다 오차가 작음
  // a 는 계산 중에 값이 망가짐, 풀 수 없으면 false
  private static bool Solve(double[,] a, double[] b)
  {
      int size = b.Length;

      for (int col = 0 ; col < size ; col++)
    {
        // 피벗: 이 열에서 나누는 수
        // 0에 가까운 수로 나누면 오차가 커지므로 절댓값이 가장 큰 행을 위로 올려서 씀
        int pivot = col;
        for (int r = col + 1 ; r < size ; r++)
      {
          if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col]))
        {
            pivot = r;
        }
      }
      // 가장 큰 값도 거의 0이면 식들이 서로 겹쳐서 해를 하나로 정할 수 없음
      if (Math.Abs(a[pivot, col]) < 1e-12)
      {
          return false;
      }

      // 행을 맞바꿀 때는 오른쪽 변 b 도 같이 바꿔야 식이 그대로 유지됨
      // (x, y) = (y, x) 는 임시 변수 없이 두 값을 맞바꾸는 문법
      if (pivot != col)
      {
          for (int c = 0 ; c < size ; c++)
        {
            (a[col, c], a[pivot, c]) = (a[pivot, c], a[col, c]);
        }
        (b[col], b[pivot]) = (b[pivot], b[col]);
      }

      // 아래 행들의 이 열을 0 으로 만듦
      // factor: 아래 행의 이 열 값이 피벗의 몇 배인지
      // 피벗 행에 factor 를 곱해 빼면 이 열이 0 이 됨
      for (int r = col + 1 ; r < size ; r++)
      {
          double factor = a[r, col] / a[col, col];
          for (int c = col ; c < size ; c++)
        {
            a[r, c] -= factor * a[col, c];
        }
        b[r] -= factor * b[col];
      }
    }

    // 아래에서부터 대입함
    // 맨 아래 식은 미지수가 하나뿐이라 바로 풀리고, 그 값을 위 식에 넣어 하나씩 올라감
    // sum: 오른쪽 변에서 이미 구한 미지수들의 몫을 뺀 나머지
    // 이것을 이 행의 대각선 값으로 나누면 이 행의 미지수
    for (int r = size - 1; r >= 0 ; r--)
    {
        double sum = b[r];
        for (int c = r + 1 ; c < size ; c++)
      {
          sum -= a[r, c] * b[c];
      }
      b[r] = sum / a[r, r];
    }
    return true;
  }
}


// ====================================================================================================
// [설계 메모]
// ====================================================================================================
// 1. 무엇을 최소화하는가 (sklearn 과 같은 문제를 푸는 법)
//    sklearn LogisticRegression(C=1.0) 은 L(w) = 1/2 * (계수 제곱합) + C * (로그 손실 합) 을 최소화함
//    로그 손실은 정답이 1인데 확률을 낮게 줬거나 정답이 0인데 높게 줬을 때 크게 커지는 값
//    C 가 작을수록 앞 항(규제)의 비중이 커서 계수가 작아짐, 코드에서는 reg = 1 / C 로 같은 효과를 냄
//    절편에는 규제를 붙이지 않음 (sklearn 도 같음), 이 점이 다르면 계수가 어긋남
//
// 2. 뉴턴법으로 푸는 이유
//    L 은 아래로 볼록해서 최저점이 하나뿐이므로 기울기가 0이 되는 곳을 찾으면 됨
//    기울기와 곡선의 휘어짐(헤세 행렬)을 함께 보고 한 번에 크게 이동하므로 이 데이터에서는 10회 안팎에 끝남
//    경사하강법은 학습률을 직접 정해야 하고 수백 번 반복해야 해서 쓰지 않음
//    역행렬을 직접 구하지 않고 H * step = grad 를 가우스 소거법(Solve)으로 풀어 오차를 줄임
//
// 3. 표준화와 누수
//    평균과 표준편차는 학습 고객에서만 구하고, 시험 고객도 학습 때 저장한 값으로 표준화함
//    시험 고객의 통계를 쓰면 시험 고객의 정보가 학습에 새어 들어가는 누수가 생김
//    표준편차는 n 으로 나눔 (StandardScaler 방식), 피처를 만들 때 쓴 n-1 과 다르니 혼동하지 말 것
//    표준편차가 0이면(모든 고객이 같은 값) 0으로 나누지 않도록 1로 둠
//
// 4. 정답 CSV 의 계수가 정확한 값이 아니었음 (C 단계에서 겪은 것을 C# 에서 다시 확인함)
//    reference_coefficients.csv 는 sklearn 기본 허용 오차(1e-4)에서 30회 만에 멈춘 값임
//    tol=1e-12 로 끝까지 수렴시키면 117회가 걸리고, 이 값과 C# 결과는 계수 3.16e-06 이내로 같음
//    tenure, active_span, gap_* 처럼 서로 비슷한 피처들은 일찍 멈추면 계수 차이가 크게 남음
//    그래서 시험은 끝까지 수렴한 값과는 엄격하게(1e-5), 정답 CSV 의 확률과는 느슨하게(1e-2) 비교함
//
// 5. C 와 달라진 점, 그리고 남은 한계
//    C 는 실패하면 printf 후 0을 돌려줬지만 C# 은 예외를 던져 이유가 호출한 쪽까지 전달됨
//    C 는 행렬 크기를 컴파일할 때 고정(PARAMS)했지만 C# 은 FeatureNames.All.Count 로 실행할 때 정함
//    IClassifier 로 약속만 분리해서 이후 MLP 와 교차검증이 같은 틀을 쓸 수 있게 함
//    한계: 행렬 풀이(Solve)는 여전히 직접 구현함, 수렴하지 못하거나 풀 수 없으면 예외로만 알려줌
// ====================================================================================================