#include "logistic.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/* 학습으로 찾아야 하는 값의 개수 */
/* 피처마다 계수 1개씩(20개) + 절편 1개 = 21개 */
/* 헤세 행렬은 21 x 21, 기울기는 길이 21 */
#define PARAMS (FEATURE_COUNT + 1)


/* 뉴턴법 반복 횟수의 상한 */
/* 보통 10회 안팎에 끝나므로 100회까지 가면 수렴하지 못하는 상황으로 봄 */
#define MAX_ITERATIONS 100

/* 한 번 반복에서 계수가 가장 크게 움직인 거리가 이 값보다 작으면 멈춤 */
/* 거의 안 움직인다는 것은 이미 최적점에 도착했다는 뜻 */
#define STEP_TOLERANCE 1e-10


/* 시그모이드 : 어떤 실수 z 든 0과 1 사이의 확률로 바꿈 */
/* sigmoid(z) = 1 / (1 + e^-z), z 가 크면 1에 가깝고 z 가 작으면 0에 가까움 */
/* z 가 아주 큰 음수이면 exp(-z) 가 너무 커져 넘치므로 부호에 따라 식을 나누도록 작성함 */
/* z < 0 일 때 e^z / (1 + e^z) 는 위 식과 수학적으로 같고, e^z 는 0에 가까운 작은 수라 안전함 */
static double sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + exp(-z));
    }
    double e = exp(z);
    return e / (1.0 + e);
}


/* 학습 고객들로 피처별 평균(mean)과 표준편차(scale)를 구해 model 에 저장함 */
/* table : 피처 전체 표, rows : 학습에 쓸 고객의 위치 목록, n : 학습 고객 수 */
/* 표준화 값 z = (x - mean) / scale 로, 평균 0 표준편차 1 의 값으로 바뀜 */
/* 모집단 표준편차이므로 n 으로 나눔 (pandas 의 n-1 과 다르며 StandardScaler 와 같은 방식) */
/* 표준편차가 0이면(모든 고객이 같은 값) 0으로 나누지 않도록 1로 둠, StandardScaler 와 동일함 */
/* 피처 단위가 제각각이라 표준화 하지 않으면 규제가 불공평해지고 행렬이 불안정해질 위험 있음! */
static void compute_scaler(const FeatureTable *table, const size_t *rows, size_t n, LogisticModel *model) {
    for (int k = 0 ; k < FEATURE_COUNT ; k++) {
        /* k번째 피처의 평균 : 학습 고객의 값을 모두 더해 n 으로 나눔 */
        double sum = 0.0;
        for (size_t i = 0 ; i < n ; i++) {
            sum += table->items[rows[i]].x[k];
        }
        double mean = sum / (double)n;

        /* 편차(값 - 평균)의 제곱을 모두 더하고 n 으로 나눈 뒤 제곱근을 씌우면 표준편차 */
        double sq = 0.0;
        for (size_t i = 0 ; i < n ; i++) {
            double d = table->items[rows[i]].x[k] - mean;
            sq += d * d;
        }
        double scale = sqrt(sq / (double)n);
        if (scale == 0.0) {
            scale = 1.0;
        }
        model->mean[k] = mean;
        model->scale[k] = scale;
    }
}


/* 가우스 소거법 - 연립방정식 풀이를 사용함 */
/* 부분 피벗 가우스 소거법으로 A x = b 를 풀어 x를 b 자리에 덮어씀 */
/* a : 21 x 21 계수 행렬 (계산 중에 값이 망가짐), b : 길이 21, 풀이가 끝나면 해 x 로 바뀜 */
/* 돌려주는 값: 1 = 성공, 0 = 풀 수 없음 */
/* 뉴턴법 한 번마다 H * step = grad 라는 연립방정식을 풀어야하는데 C에는 numpy 가 없어 직접 풀어야 함 */
/* 방법 : 위쪽 행을 이용해 아래 행의 해당 열을 0으로 만들어 삼각형 모양으로 바꾼 뒤(소거), 맨 아래 식부터 차례로 대입함 */
static int solve(double a[PARAMS][PARAMS], double b[PARAMS]) {
    for (int col = 0 ; col < PARAMS ; col++) {

      /* 이 열에서 절댓값이 가장 큰 행을 위로 올림 */
      /* 0에 가까운 수로 나누면 오차가 커지므로 가장 큰 수를 나누는 수(피벗)로 삼기 위함 */
      int pivot = col;
      for (int r = col + 1 ; r < PARAMS ; r++) {
          if (fabs(a[r][col]) > fabs(a[pivot][col])) {
              pivot = r;
          }
      }
      /* 가장 큰 값도 거의 0이면 식들이 서로 겹쳐서 해를 하나로 정할 수 없음 */
      if (fabs(a[pivot][col]) < 1e-12) {
          return 0;
      }
      /* 행을 맞바꿀 때는 오른쪽 변 b 도 같이 바꿔야 식이 그대로 유지됨 */
      if (pivot != col) {
          for (int c = 0 ; c < PARAMS ; c++) {
              double t = a[col][c];
              a[col][c] = a[pivot][c];
              a[pivot][c] = t;
          }
          double t = b[col];
          b[col] = b[pivot];
          b[pivot] = t;
      }

      /* 아래 행들의 이 열을 0으로 만듦 */
      /* factor : 아래 행의 이 열 값이 피벗의 몇 배인지, 피벗 행에 factor 를 곱해 빼면 그 열이 0이 됨 */
      for (int r = col + 1 ; r < PARAMS ; r++) {
          double factor = a[r][col] / a[col][col];
          for (int c = col ; c < PARAMS ; c++) {
              a[r][c] -= factor * a[col][c];
          }
          b[r] -= factor * b[col];
      }
    }

    /* 아래에서부터 대입 */
    /* 맨 아래 식은 미지수가 하나뿐이라 바로 풀리고, 그 값을 위 식에 넣어 하나씩 올라감 */
    /* sum : 오른쪽 변에서 이미 구한 미지수들의 몫을 뺀 나머지 */
    for (int r = PARAMS - 1 ; r >= 0 ; r--) {
        double sum = b[r];
        for (int c = r + 1 ; c < PARAMS ; c++) {
            sum -= a[r][c] * b[c];
        }
        b[r] = sum / a[r][r];
    }
    return 1;
}


/* 학습 고객의 피처로 계수 20개와 절편을 찾음 (뉴턴법) */
/* 찾는 값 : 아래 함수 L(w) 를 가장 작게 만드는 w */
/*   L(w) = 1/2 * (계수들의 제곱합) + C * (고객별 로그 손실의 합) */
/*   앞 항은 계수가 커지는 것을 막는 규제, 뒤 항은 예측이 틀린 정도 */
/* 하단 설계 메모 1, 2 참고 */
int logistic_fit(const FeatureTable *table, const size_t *rows, size_t n, double c_value, LogisticModel *model) {
    if (n == 0) {
        printf("학습할 고객이 없음\n");
        return 0;
    }
    compute_scaler(table, rows, n, model);


    /* 표준화한 피처를 한 번만 계산해 둠 */
    /* n * 20 */
    /* 행 순서대로 이어 붙인 1차원 배열 */
    /* 고객 i 의 k번째 값은 z[i * 20 + k] */
    /* 반복할 때마다 계산하면 같은 나눗셈을 10번 되풀이하기 때문에 미리 만들어 둠 */
    double *z = malloc(n * FEATURE_COUNT * sizeof(double));
    if (z == NULL) {
        printf("메모리 할당 실패\n");
        return 0;
    }
    for (size_t i = 0 ; i < n ; i++) {
        for (int k = 0 ; k < FEATURE_COUNT ; k++) {
            z[i * FEATURE_COUNT + k] = (table->items[rows[i]].x[k] - model->mean[k]) / model->scale[k];
        }
    }

    /* w[0]부터 w[19]는 계수를 나타냄 */
    /* w[20]은 절편 */
    /* 처음엔 모두 0으로 시작함 (모든 고객의 확률이 0.5 인 상태) */
    /* reg : 규제 세기, sklearn 의 C 가 1이면 reg 도 1, C 가 작을수록 reg 가 커져 계수를 더 억누름 */
    /* converged : 수렴해서 멈췄는지 기록함 */
    double w[PARAMS] = {0.0};
    double reg = 1.0 / c_value;
    int converged = 0;

    for (int iter = 1 ; iter <= MAX_ITERATIONS ; iter++) {
        /* grad : 기울기, L 을 w 의 각 값으로 미분한 것 (길이 21) */
        /*        각 값을 키우면 L 이 얼마나 커지는지를 나타내며, 최적점에서는 모두 0 */
        /* hess : 헤세 행렬, 기울기가 w 에 따라 얼마나 변하는지 (21 x 21) */
        /*        곡선이 얼마나 휘어 있는지 알려줘서 한 번에 얼마나 움직일지 정해줌 */
        double grad[PARAMS] = {0.0};
        double hess[PARAMS][PARAMS] = {{0.0}};

        /* 고객 한 명씩 기울기와 헤세 행렬에 더해 감 */
        for (size_t i = 0 ; i < n ; i++) {
            /* zi : 이 고객의 표준화된 피처 20개 (z 배열 안의 이 고객 시작 위치) */
            const double *zi = &z[i * FEATURE_COUNT];

            /* s : 현재 w 로 계산한 점수 = 절편 + 계수 * 피처의 합 */
            double s = w[FEATURE_COUNT];
            for (int k = 0 ; k < FEATURE_COUNT ; k++) {
                s += w[k] * zi[k];
            }

            /* p : 점수를 확률로 바꾼 값 (이 고객이 이탈할 확률) */
            /* err : 확률과 실제 정답(이탈 1, 유지 0)의 차이, 기울기 식에서 (p - y) 에 해당함 */
            /* weight : p * (1 - p), 헤세 행렬에 곱해지는 값 */
            /*          확률이 0.5 에 가까워 판단이 애매한 고객일수록 커지고, 0이나 1에 가까우면 작아짐 */
            double p = sigmoid(s);
            double err = p -(double)table->items[rows[i]].churn;
            double weight = p * (1.0 - p);

            /* 절편 칸은 값이 1인 피처가 하나 더 있는 것과 동일함 */
            /* za, zb : a번째, b번째 칸의 피처 값 (절편 칸이면 1) */
            /* 기울기의 a번째 값 += err * za */
            /* 헤세 행렬의 (a, b) 칸 += weight * za * zb */
            for (int a = 0 ; a < PARAMS ; a++){
                double za = 1.0;
                if (a < FEATURE_COUNT) {
                    za = zi[a];
                }
                grad[a] += err * za;
                /* (a, b) 와 (b, a) 는 같은 값이라 b <= a 인 아래 삼각형만 계산함 */
                for (int b = 0 ; b <= a ; b++) {
                    double zb = 1.0;
                    if (b < FEATURE_COUNT) {
                        zb = zi[b];
                    }
                    hess[a][b] += weight *za * zb;
                }
            }
        }

        /* 아래 삼각형만 채웠으므로 위 삼각형에 복사함 */
        for (int a = 0 ; a < PARAMS ; a++) {
            for (int b = a + 1 ; b < PARAMS ; b++) {
                hess[a][b] = hess[b][a];
            }
        }

        /* 규제 항 */
        /* 계수에만 붙이고 절편에는 붙이지 않음 */
        /* L 의 앞 항 1/2 * reg * w^2 을 w 로 미분하면 reg * w, 한 번 더 미분하면 reg */
        /* 그래서 기울기에는 reg * w 를, 헤세 행렬의 대각선에는 reg 를 더함 */
        for (int k = 0 ; k < FEATURE_COUNT ; k++) {
            grad[k] += reg * w[k];
            hess[k][k] += reg;
        }

        /* 뉴턴 방향 */
        /* H step = grad 를 풀어 w에서 빼는 방식으로 계산함 */
        /* solve 가 끝나면 grad 자리에 step(이번에 움직일 방향과 크기)이 들어 있음 */
        if (!solve(hess, grad)) {
            printf("헤세 행렬을 풀 수 없음 (반복 %d)\n", iter);
            free(z);
            return 0;
        }

        /* w 를 step 만큼 이동시키고, 가장 크게 움직인 값(max_step)을 기록함 */
        double max_step = 0.0;
        for (int a = 0 ; a < PARAMS ; a++) {
            w[a] -= grad[a];
            if (fabs(grad[a]) > max_step) {
                max_step = fabs(grad[a]);
            }
        }
        model->iterations = iter;
        if (max_step < STEP_TOLERANCE) {
            converged = 1;
            break;
        }
    }
    free(z);

    if (!converged) {
        printf("%d회 안에 수렴하지 않음\n", MAX_ITERATIONS);
        return 0;
    }

    /* 결과를 모델에 옮겨 담음 */
    for (int k = 0 ; k < FEATURE_COUNT ; k++) {
        model->coef[k] = w[k];
    }
    model->intercept = w[FEATURE_COUNT];
    return 1;
}


/* 고객 한 명의 이탈 확률 */
/* 원래 값을 학습 때 구한 평균과 표준편차로 표준화한 뒤 계수를 곱해 더하고 시그모이드를 씌움 */
/* 시험 고객도 학습 고객의 평균과 표준편차를 쓰는 것이 중요함 (시험 고객의 통계를 쓰면 누수) */
double logistic_predict(const LogisticModel *model, const CustomerFeatures *customer) {
    double s = model->intercept;
    for (int k = 0 ; k < FEATURE_COUNT ; k++) {
        s += model->coef[k] * (customer->x[k] - model->mean[k]) / model->scale[k];
    }
    return sigmoid(s);
}


/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 무엇을 최소화하는가 (sklearn 과 같은 문제를 푸는 법) */
/*    sklearn LogisticRegression(C=1.0) 은 L(w) = 1/2 * (계수 제곱합) + C * (로그 손실 합) 을 최소화함 */
/*    로그 손실은 정답이 1인데 확률을 낮게 줬거나, 정답이 0인데 높게 줬을 때 크게 커지는 값 */
/*    C 가 작을수록 앞 항(규제)의 비중이 커서 계수가 작아짐, 코드에서는 reg = 1 / C 로 같은 효과를 냄 */
/*    절편에는 규제를 붙이지 않음 (sklearn 도 같음), 이 점이 다르면 계수가 어긋남 */
/* */
/* 2. 뉴턴법으로 푸는 이유와 수식 */
/*    L 은 아래로 볼록해서 최저점이 하나뿐이므로 기울기가 0이 되는 곳을 찾으면 됨 */
/*    기울기   : 고객마다 (p - y) * x 를 더하고, 계수에는 reg * w 를 더함 */
/*    헤세 행렬: 고객마다 p(1-p) * x * x 의 전치를 더하고, 계수 대각선에는 reg 를 더함 */
/*    뉴턴법은 w 를 w - (헤세 행렬)의 역 * 기울기 로 옮기는 것을 되풀이함 */
/*    곡선의 휘어짐까지 고려해서 한 번에 크게 이동하므로 이 데이터에서는 10회 안팎에 끝남 */
/*    경사하강법은 학습률을 직접 정해야 하고 수백 번 반복해야 해서 쓰지 않음 */
/*    역행렬을 직접 구하지 않고 H * step = grad 를 가우스 소거법으로 풀어 오차를 줄임 */
/* */
/* 3. 절편과 표준화 */
/*    절편은 모든 고객에게 값이 1인 피처가 하나 더 있는 것으로 보면 계수와 같은 틀(21칸)로 계산 가능 */
/*    평균과 표준편차는 학습 고객에서만 구해 시험 고객에게도 그대로 씀 (파이썬에서 Pipeline 안에 넣은 이유와 같음) */
/*    표준편차는 n 으로 나눔 (StandardScaler 방식), 피처를 만들 때 쓴 n-1 과 다르니 혼동하지 말 것 */
/* */
/* 4. 정답 CSV 의 계수가 정확한 값이 아니었음 (이번 단계에서 직접 겪음) */
/*    처음에 reference_coefficients.csv 와 비교했더니 계수가 최대 0.016, 확률이 최대 0.008 어긋남 */
/*    sklearn 을 tol=1e-12 로 끝까지 수렴시키니(117회) C 결과와 소수 6자리까지 같았음 */
/*    정답 CSV 는 기본 허용 오차 1e-4 에서 30회 만에 멈춘 값이라 최적점에서 조금 떨어져 있었음 */
/*    tenure, active_span, gap_* 처럼 서로 비슷한 피처들은 한쪽이 커지고 다른 쪽이 줄어도 예측이 거의 같아서 */
/*    일찍 멈추면 이 계수들의 차이가 크게 남음 */
/*    그래서 시험은 끝까지 수렴한 값(tools/expected_values.py 출력)과는 엄격하게, 정답 CSV 와는 느슨하게 비교함 */
/* */
/* 5. 한계 (이번 단계에서 직접 겪음) */
/*    행렬 계산 라이브러리가 없어 가우스 소거법(solve)을 직접 만들어야 했음 */
/*    행렬 크기(PARAMS)를 컴파일할 때 고정하므로 피처 개수를 바꾸면 헤더의 enum 을 고치고 다시 빌드해야 함 */
/*    피처 사이 관계가 너무 강해 헤세 행렬이 거의 특이해지면 solve 가 실패하는데, 이 데이터에서는 규제 항이 대각선에 더해져 안전함 */
/* ==================================================================================================== */
