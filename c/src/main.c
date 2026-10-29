/* 이탈 예측 파이프라인 (C 구현) */
/* 10 단계 : 전체 연결 + 파이썬 결과와 비교 */
/* 거래 CSV -> 이탈 라벨 -> 피처 20개 -> 로지스틱 회귀 -> 평가 -> 기대이익 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv.h"
#include "datetime.h"
#include "features.h"
#include "labels.h"
#include "logistic.h"
#include "metrics.h"
#include "profit.h"
#include "returns.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif


#define EXPORT_DIR "data/processed/export_for_c/"

/* 관측창: 이 시각 미만 */
/* 예측창: 이 시각 이상 ~ 끝 시각 미만 */
#define CUTOFF_TEXT "2011-06-10 00:00:00"
#define LABEL_END_TEXT "2011-12-10 00:00:00"

/* 규제 세기의 역수 */
/* sklearn LogisticRegression 의 기본값 C = 1.0 */
#define REGULARIZATION_C 1.0

/* 임계값 0.5 : 확률이 0.5 이상이면 이탈로 예측함 (sklearn 기본값) */
#define DEFAULT_THRESHOLD 0.5

/* 가정 : 공헌이익률, 쿠폰 1장 원가(파운드), 이탈 예정 고객이 쿠폰을 받고 돌아올 확률 */
/* 파이썬 analysis/a5_profit.py 의 MARGIN, COUPON_COST, RESPONSE 와 같은 값 */
#define MARGIN 0.30
#define COUPON_COST 5.00
#define RESPONSE 0.10

/* 피처는 숫자 하나하나가 정답과 같아야 하므로 엄격하게 봄 (상대 오차) */
#define FEATURE_TOLERANCE 1e-9


/* 파이썬 정답 요약 (reference_summary.csv) */
typedef struct {
    double test_auc;
    double intercept;
    double n_train;
    double n_test;
    double churn_rate_train;
    double churn_rate_test;
    double r_retained_median;
    double value_tp;
    double value_fp;
    double best_threshold;
    double best_profit;
    double profit_at_050;
} Summary;



/* reference_summary.csv 를 읽어 summary 에 채움 */
/* 파일은 "키,값" 한 줄씩이며, 필요한 키가 하나라도 없으면 실패 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
static int summary_load(Summary *summary) {
    /* 키 이름과 그 값을 담을 자리를 짝지어 둔 표 */
    struct {
        const char *key;
        double *target;
        int found;
    } entries[] = {
        {"test_auc", &summary->test_auc, 0},
        {"intercept", &summary->intercept, 0},
        {"n_train", &summary->n_train, 0},
        {"n_test", &summary->n_test, 0},
        {"churn_rate_train", &summary->churn_rate_train, 0},
        {"churn_rate_test", &summary->churn_rate_test, 0},
        {"r_retained_median", &summary->r_retained_median, 0},
        {"value_tp", &summary->value_tp, 0},
        {"value_fp", &summary->value_fp, 0},
        {"best_threshold", &summary->best_threshold, 0},
        {"best_profit", &summary->best_profit, 0},
        {"profit_at_0.50", &summary->profit_at_050, 0},
    };
    int entry_count = (int)(sizeof(entries) / sizeof(entries[0]));

    FILE *fp = fopen(EXPORT_DIR "reference_summary.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_summary.csv\n");
        return 0;
    }
    char line[256];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, f, CSV_MAX_FIELDS);
        for (int i = 0; i < entry_count; i++) {
            if (strcmp(f[0], entries[i].key) == 0) {
                *entries[i].target = atof(f[1]);
                entries[i].found = 1;
            }
        }
    }
    fclose(fp);

    for (int i = 0; i < entry_count; i++) {
        if (!entries[i].found) {
            printf("reference_summary.csv 에 키가 없음: %s\n", entries[i].key);
            return 0;
        }
    }
    return 1;
}


/* reference_split.csv 를 읽어 학습, 시험 고객의 위치를 채움 */
/* 파이썬이 sklearn 의 train_test_split 으로 나눈 결과를 그대로 씀 */
/* C 에서는 같은 무작위 분할을 재현할 수 없음 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
static int read_split(const FeatureTable *features, size_t *train, size_t *n_train,
                      size_t *test, size_t *n_test) {
    FILE *fp = fopen(EXPORT_DIR "reference_split.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_split.csv\n");
        return 0;
    }
    char line[256];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));

    *n_train = 0;
    *n_test = 0;
    size_t row = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, f, CSV_MAX_FIELDS);
        if (row >= features->count || features->items[row].customer_id != atoi(f[0])) {
            printf("고객 순서가 다름: %zu행\n", row);
            fclose(fp);
            return 0;
        }
        if (strcmp(f[1], "train") == 0) {
            train[*n_train] = row;
            (*n_train)++;
        } else {
            test[*n_test] = row;
            (*n_test)++;
        }
        row++;
    }
    fclose(fp);
    return row == features->count;
}



/* 만든 피처를 reference_features.csv 와 고객별, 피처별로 비교함 */
/* max_error : 20개 피처 전체에서 가장 큰 상대 오차, worst : 그 피처의 번호 */
/* wrong_labels : 이탈, 신규 라벨이 정답과 다른 고객 수 */
/* 돌려주는 값: 1 = 비교 완료, 0 = 파일이 없거나 고객 순서가 다름 */
static int compare_features(const FeatureTable *features, double *max_error, int *worst,
                            long long *wrong_labels) {
    FILE *fp = fopen(EXPORT_DIR "reference_features.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_features.csv\n");
        return 0;
    }
    char line[2048];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));

    *max_error = 0.0;
    *worst = 0;
    *wrong_labels = 0;
    size_t row = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, f, CSV_MAX_FIELDS);
        if (row >= features->count || features->items[row].customer_id != atoi(f[0])) {
            printf("고객 순서가 다름: %zu행\n", row);
            fclose(fp);
            return 0;
        }
        const CustomerFeatures *c = &features->items[row];
        for (int k = 0; k < FEATURE_COUNT; k++) {
            double expected = atof(f[k + 1]);
            double error = fabs(c->x[k] - expected) / fmax(1.0, fabs(expected));
            if (error > *max_error) {
                *max_error = error;
                *worst = k;
            }
        }
        if (c->churn != atoi(f[FEATURE_COUNT + 1]) || c->is_new != atoi(f[FEATURE_COUNT + 2])) {
            (*wrong_labels)++;
        }
        row++;
    }
    fclose(fp);
    return row == features->count;
}


/* 단계에 걸린 시간(초) */
static double seconds_since(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}


/* 파이썬 값과 C 값을 한 줄로 비교해 출력함 */
/* 차이 = (C - 파이썬) / max(1, |파이썬|) */
/* 1보다 작은 값은 절대 차이, 큰 값은 상대 차이 */
/* 허용 오차 이하이면 통과 */
/* 돌려주는 값: 불일치 1, 일치 0 */
static int compare_row(const char *name, double python_value, double c_value, double tolerance) {
    double error = fabs(c_value - python_value) / fmax(1.0, fabs(python_value));
    const char *verdict = "OK";
    int mismatch = 0;
    if (error > tolerance) {
        verdict = "불일치";
        mismatch = 1;
    }
    printf("  %-18s %14.6f %14.6f %10.2e %8.0e  %s\n", name, python_value, c_value, error, tolerance, verdict);
    return mismatch;
}



int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    /* total_start : 프로그램 전체 시간의 시작점 */
    /* start : 단계마다 다시 찍어 쓰는 시작점 */
    clock_t total_start = clock();
    clock_t start;

    printf("=== 이탈 예측 파이프라인 (C 구현) ===\n\n");

    /* 파이썬 정답 요약을 먼저 읽어 둠 */
    /* 맨 끝 비교 표에서 사용함 */
    /* 파일이 없거나 키가 빠졌으면 여기서 바로 끝냄 */
    Summary summary;
    if (!summary_load(&summary)) {
        return 1;
    }


    /* 1. 데이터 읽기 */
    /* 거래 77만 건과 반품 1.8만 건을 메모리의 배열(tx, returns)로 올림 */
    /* 이 두 표가 이후 모든 단계의 입력을 나타내도록 작성함 */
    /* 맨 끝에서 free 로 돌려줌 */
    start = clock();
    TransactionTable tx;
    if (!transactions_load(EXPORT_DIR "clean_transactions.csv", &tx)) {
        return 1;
    }
    ReturnTable returns;
    if (!returns_load(EXPORT_DIR "returns.csv", &returns)) {
        return 1;
    }
    printf("[데이터] 거래 %zu건, 반품 %zu건 (%.2f초)\n", tx.count, returns.count, seconds_since(start));


    /* 2. 이탈 라벨 */
    /* cutoff : 관측창과 예측창을 가르는 시각 */
    /* label_end : 예측창의 끝 시각 */
    /* 둘 다 1970년부터 지난 초를 나타냄 */
    /* 관측창에 구매한 고객마다 "예측창에 다시 구매했는가" 를 보고 이탈(1)/유지(0) 를 정함 */
    long long cutoff, label_end;
    dt_parse(CUTOFF_TEXT, &cutoff);
    dt_parse(LABEL_END_TEXT, &label_end);

    start = clock();
    LabelTable labels;
    if (!labels_build(&tx, cutoff, label_end, &labels)) {
        return 1;
    }

    /* churned : 이탈 고객 수 */
    /* new_customers : 신규 고객 수 (churn, is_new 가 0 또는 1 이라 더하면 개수가 됨) */
    long long churned = 0;
    long long new_customers = 0;
    for (size_t i = 0 ; i < labels.count ; i++) {
        churned += labels.items[i].churn;
        new_customers += labels.items[i].is_new;
    }
    printf("[라벨] 고객 %zu명, 이탈 %lld명 (%.1f%%), 신규 %lld명 (%.2f초)\n", labels.count, churned,
            100.0 * (double)churned / (double)labels.count, new_customers, seconds_since(start));

    
    /* 3. 피처 20개 */
    /* 고객마다 관측창(cutoff 이전) 거래로 피처 20개를 계산함 */
    /* 비교 시간까지 재면 파일 읽는 시간이 섞이므로 만드는 시간만 먼저 따로 저장해 둠 */
    start = clock();
    FeatureTable features;
    if (!features_build(&tx, &returns, &labels, cutoff, &features)) {
        return 1;
    }
    double feature_seconds = seconds_since(start);

    /* feature_error : 20개 피처 전체에서 가장 큰 상대 오차 */
    /* worst_feature : 그 피처의 번호 */
    /* wrong_labels : 이탈, 신규 라벨이 정답과 다른 고객 수 */
    double feature_error;
    int worst_feature;
    long long wrong_labels;
    if (!compare_features(&features, &feature_error, &worst_feature, &wrong_labels)) {
        return 1;
    }
    printf("[피처] %zu명 x %d개 (%.2f초)\n", features.count, FEATURE_COUNT, feature_seconds);
    printf("       정답 대비 최대 상대 오차 %.2e (%s), 라벨 불일치 %lld명\n", feature_error,
           FEATURE_NAMES[worst_feature], wrong_labels);

    
    /* 4. 학습, 시험 분할 */
    /* train, test : features 표에서 몇 번째 고객인지를 적어 둔 위치 목록 */
    /* 고객 자체를 복사하지 않음 */
    /* 한 명이 어느 쪽에 들어갈지 몰라 둘 다 전체 고객 수만큼 확보함 */
    size_t *train = malloc(features.count * sizeof(size_t));
    size_t *test = malloc(features.count * sizeof(size_t));
    if (train == NULL || test == NULL) {
        printf("메모리 할당 실패\n");
        return 1;
    }
    size_t n_train, n_test;
    if (!read_split(&features, train, &n_train, test, &n_test)) {
        return 1;
    }

    /* train_churn : 학습 고객 중 이탈 고객 수 */
    /* 마지막 비교 표에서 이탈률(정답 요약과 비교)을 구할 때 사용할 것 */
    long long train_churn = 0;
    for (size_t i = 0 ; i < n_train ; i++) {
        train_churn += features.items[train[i]].churn;
    }
    printf("[분할] 학습 %zu명, 시험 %zu명\n", n_train, n_test);



    /* 5. 로지스틱 회귀 학습 */
    /* 학습 고객(train 위치 목록)만으로 표준화 기준(평균, 표준편차)과 계수를 구함 */
    /* 시험 고객은 쓰면 시험 결과가 부풀려지므로 여기서 사용하지 않음 */
    start = clock();
    LogisticModel model;
    if (!logistic_fit(&features, train, n_train, REGULARIZATION_C, &model)) {
        return 1;
    }
    printf("[학습] 뉴턴법 %d회 반복, 절편 %+.4f (%.2f초)\n", model.iterations, model.intercept,
           seconds_since(start));
    

    
    /* 6. 시험 고객 예측 */
    /* y : 시험 고객의 실제 정답 (이탈 1, 유지 0) */
    /* prob : 학습한 모델이 계산한 이탈 확률 */
    /* i 번째 칸이 시험 고객 i 번째 한 명에 해당하며 두 배열의 칸 순서가 같아야 함 */
    /* 파이썬이 저장한 확률을 읽지 않고 C 모델이 직접 계산한 확률을 씀 */
    int *y = malloc(n_test * sizeof(int));
    double *prob = malloc(n_test * sizeof(double));
    if (y == NULL || prob == NULL) {
        printf("메모리 할당 실패\n");
        return 1;
    }
    /* test_churn : 시험 고객 중 이탈 고객 수 (이탈률 비교용) */
    long long test_churn = 0;
    for (size_t i = 0 ; i < n_test ; i++) {
        const CustomerFeatures *c = &features.items[test[i]];
        y[i] = c->churn;
        prob[i] = logistic_predict(&model, c);
        test_churn += c->churn;
    }



    /* 7. 평가 지표 */
    /* 확률이 0.5 이상이면 이탈로 예측했을 때의 지표 7개와 혼동행렬을 한 번에 구함 */
    /* AUC, PR-AUC, Brier 는 임계값과 무관하고 나머지는 임계값 0.5 기준임 */
    Metrics m;
    if (!metrics_evaluate(y, prob, n_test, DEFAULT_THRESHOLD, &m)) {
        return 1;
    }
    printf("\n[평가] 시험 고객 %zu명, 임계값 %.2f\n", n_test, DEFAULT_THRESHOLD);
    printf("  AUC %.4f / PR-AUC %.4f / accuracy %.4f / precision %.4f / recall %.4f / F1 %.4f / Brier %.4f\n",
           m.auc, m.pr_auc, m.accuracy, m.precision, m.recall, m.f1, m.brier);
    printf("  혼동행렬  TN %lld, FP %lld, FN %lld, TP %lld\n", m.cm.tn, m.cm.fp, m.cm.fn, m.cm.tp);



    /* 8. 기대이익 */
    /* retained : 유지 고객이 예측창에서 쓴 금액의 중앙값, 평균, 인원 (데이터에서 잰 값) */
    RetainedRevenue retained;
    if (!profit_measure_retained(&tx, &labels, cutoff, label_end, &retained)) {
        return 1;
    }

    /* assumptions : 마진, 쿠폰 원가, 반응률 (가정) */
    /* values : 쿠폰 1장을 보냈을 때의 이익 (맞게 보냄 / 헛되이 보냄) */
    ProfitAssumptions assumptions = {MARGIN, COUPON_COST, RESPONSE};
    ProfitValues values = profit_values(&assumptions, retained.median);

    /* curve : 임계값 0.05 ~ 0.95 (19개) 마다 쿠폰을 보냈을 때의 발송 수, 맞춘 수, 기대이익 */
    /* best : 기대이익이 가장 큰 칸의 번호 */
    /* base : 기본 임계값 0.50 의 칸 번호 (0.05 * (9 + 1) = 0.50) */
    ProfitPoint curve[PROFIT_THRESHOLDS];
    profit_curve(y, prob, n_test, &values, curve);
    int best = profit_best_index(curve);
    int base = 9;
    printf("\n[기대이익] 마진 %.0f%%, 쿠폰 원가 %.2f, 반응률 %.0f%%, 유지 고객 매출 중앙값 %.2f\n",
           MARGIN * 100.0, COUPON_COST, RESPONSE * 100.0, retained.median);
    printf("  맞게 보낸 1건 %+.2f, 헛되이 보낸 1건 %+.2f\n", values.value_tp, values.value_fp);
    printf("  기본값 %.2f -> 발송 %lld건, 기대이익 %.2f\n", curve[base].threshold, curve[base].sent, curve[base].profit);
    printf("  최적  %.2f -> 발송 %lld건, 기대이익 %.2f (기본값보다 %.2f 많음)\n", curve[best].threshold,
           curve[best].sent, curve[best].profit, curve[best].profit - curve[base].profit);
           
    

    /* 9. 파이썬 결과와 비교 */
    /* compare_row 가 한 줄씩 출력하고 허용 오차를 넘으면 1 을 돌려줌 */
    /* 그 합이 mismatches */
    /* 표 이름은 한글이 섞이면 칸이 어긋나므로 영어로 씀 */
    /* 허용 오차는 항목마다 다름 : 개수, 중앙값은 정확히 같아야 하고 (1e-9 이하), */
    /* 계수, AUC, 이익은 정답 CSV 가 덜 수렴한 값이라 조금 느슨하게 둠 */
    /* 설계 메모 logistic.c 하단 참고 */
    printf("\n[파이썬 정답과 비교]\n");
    printf("  %-18s %14s %14s %10s %8s\n", "item", "python", "c", "diff", "tol");
    int mismatches = 0;
    mismatches += compare_row("customers_train", summary.n_train, (double)n_train, 1e-12);
    mismatches += compare_row("customers_test", summary.n_test, (double)n_test, 1e-12);
    mismatches += compare_row("churn_rate_train", summary.churn_rate_train, (double)train_churn / (double)n_train, 1e-9);
    mismatches += compare_row("churn_rate_test", summary.churn_rate_test, (double)test_churn / (double)n_test, 1e-9);
    mismatches += compare_row("intercept", summary.intercept, model.intercept, 2e-3);
    mismatches += compare_row("test_auc", summary.test_auc, m.auc, 1e-3);
    mismatches += compare_row("retained_median", summary.r_retained_median, retained.median, 1e-9);
    mismatches += compare_row("value_tp", summary.value_tp, values.value_tp, 1e-3);
    mismatches += compare_row("value_fp", summary.value_fp, values.value_fp, 1e-9);
    mismatches += compare_row("best_threshold", summary.best_threshold, curve[best].threshold, 1e-9);
    mismatches += compare_row("best_profit", summary.best_profit, curve[best].profit, 1e-2);
    mismatches += compare_row("profit@0.50", summary.profit_at_050, curve[base].profit, 1e-2);

    /* 피처 20개와 라벨은 3 단계에서 이미 비교했으므로 그 결과(최대 오차, 라벨 불일치 수)만 여기서 판정함 */
    if (feature_error > FEATURE_TOLERANCE || wrong_labels != 0) {
        printf("  피처 또는 라벨이 정답과 다름\n");
        mismatches++;
    }

    /* 쓴 메모리를 모두 돌려줌 */
    /* malloc, calloc 한 것은 free 로 짝을 맞춤 */
    free(y);
    free(prob);
    free(train);
    free(test);
    features_free(&features);
    labels_free(&labels);
    returns_free(&returns);
    transactions_free(&tx);

    /* 불일치가 하나도 없을 때만 종료 코드 0 (성공), 아니면 1 (실패) */
    /* 종료 코드를 보면 사람이 출력을 읽지 않아도 자동으로 통과 여부를 알 수 있음 */
    printf("\n전체 %.2f초, ", seconds_since(total_start));
    if (mismatches == 0) {
        printf("파이썬 결과와 모두 일치\n");
        return 0;
    }
    printf("불일치 %d건\n", mismatches);
    return 1;
}