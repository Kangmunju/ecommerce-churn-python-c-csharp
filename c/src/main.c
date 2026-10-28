/* 이탈 예측 파이프라인 (C 구현) */
/* 9 단계 : 기대이익 */

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
#include "profit.h"
#include "returns.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif


#define EXPORT_DIR "data/processed/export_for_c/"

/* 같은 확률과 같은 데이터로 계산한 값끼리 비교하므로 엄격하게 봄 (상대 오차) */
#define PROFIT_TOLERANCE 1e-9

/* 가정 : 공헌이익률, 쿠폰 1장 원가(파운드), 이탈 예정 고객이 쿠폰을 받고 돌아올 확률 */
/* 파이썬 analysis/a5_profit.py 의 MARGIN, COUPON_COST, RESPONSE 와 같은 값 */
#define MARGIN 0.30
#define COUPON_COST 5.00
#define RESPONSE 0.10

/* tools/expected_values.py 의 [유지 고객의 예측창 매출] 출력 */
#define EXPECTED_RETAINED_COUNT 2572
#define EXPECTED_RETAINED_MEDIAN 635.0100000000
#define EXPECTED_RETAINED_MEAN 1758.2904315708
#define EXPECTED_VALUE_TP 14.0503000000
#define EXPECTED_VALUE_FP -5.0000000000

/* tools/expected_values.py 의 [임계값별 이익 (reference_test_pred.csv)] 출력 */
#define EXPECTED_BEST_THRESHOLD 0.30
#define EXPECTED_BEST_PROFIT 6969.342900
#define EXPECTED_SUM_SENT 13380
#define EXPECTED_SUM_PROFIT 91884.250500


/* reference_split.csv 를 읽어 학습, 시험 고객의 위치를 채움 */
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


/* customer_id 로 피처 표에서 고객을 찾음 (표는 customer_id 오름차순) */
static const CustomerFeatures *find_customer(const FeatureTable *features, int customer_id) {
    size_t lo = 0;
    size_t hi = features->count;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (features->items[mid].customer_id < customer_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == features->count || features->items[lo].customer_id != customer_id) {
        return NULL;
    }
    return &features->items[lo];
}


/* 계산한 값이 기대값과 같은지 확인하고 결과를 한 줄로 출력함 */
/* 상대 오차(기대값이 1보다 작으면 절대 오차)가 tolerance 이하이면 통과 */
/* 돌려주는 값: 실패 1, 통과 0 */
static int check_value(const char *name, double actual, double expected, double tolerance) {
    double error = fabs(actual - expected) / fmax(1.0, fabs(expected));
    if (error <= tolerance) {
        printf("  OK   %-14s %.6f (오차 %.2e)\n", name, actual, error);
        return 0;
    }
    printf("  실패 %-14s %.6f, 기대 %.6f (오차 %.2e)\n", name, actual, expected, error);
    return 1;
}


/* 개수가 기대하는 값과 정확히 같은지 확인함 */
/* 돌려주는 값: 실패 1, 통과 0 */
static int check_count(const char *name, long long actual, long long expected) {
    if (actual == expected) {
        printf("  OK   %-14s %lld\n", name, actual);
        return 0;
    }
    printf("  실패 %-14s %lld, 기대 %lld\n", name, actual, expected);
    return 1;
}


/* 이익 곡선을 표로 출력함 */
static void print_curve(const ProfitPoint curve[PROFIT_THRESHOLDS]) {
    int best = profit_best_index(curve);
    printf("  임계값   발송    TP    FP  정밀도      기대이익\n");
    for (int k = 0; k < PROFIT_THRESHOLDS; k++) {
        const char *mark = "";
        if (k == best) {
            mark = "  <- 최대";
        }
        printf("   %.2f  %5lld %5lld %5lld   %.2f  %10.2f%s\n", curve[k].threshold, curve[k].sent,
               curve[k].tp, curve[k].fp, curve[k].precision, curve[k].profit, mark);
    }
}


int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    TransactionTable tx;
    if (!transactions_load(EXPORT_DIR "clean_transactions.csv", &tx)) {
        return 1;
    }
    ReturnTable returns;
    if (!returns_load(EXPORT_DIR "returns.csv", &returns)) {
        return 1;
    }
    long long cutoff, label_end;
    dt_parse("2011-06-10 00:00:00", &cutoff);
    dt_parse("2011-12-10 00:00:00", &label_end);

    LabelTable labels;
    if (!labels_build(&tx, cutoff, label_end, &labels)) {
        return 1;
    }
    FeatureTable features;
    if (!features_build(&tx, &returns, &labels, cutoff, &features)) {
        return 1;
    }

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
    LogisticModel model;
    if (!logistic_fit(&features, train, n_train, 1.0, &model)) {
        return 1;
    }

    /* reference_test_pred.csv 에서 시험 고객의 정답(y)과 파이썬이 계산한 확률(ref_prob)을 읽음 */
    /* 같은 고객에 대해 C 모델이 계산한 확률(my_prob)도 함께 만듦 */
    int *y = malloc(n_test * sizeof(int));
    double *ref_prob = malloc(n_test * sizeof(double));
    double *my_prob = malloc(n_test * sizeof(double));
    if (y == NULL || ref_prob == NULL || my_prob == NULL) {
        printf("메모리 할당 실패\n");
        return 1;
    }
    FILE *fp = fopen(EXPORT_DIR "reference_test_pred.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_test_pred.csv\n");
        return 1;
    }
    char line[256];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));
    size_t rows = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1 && rows < n_test) {
        csv_split(line, f, CSV_MAX_FIELDS);
        const CustomerFeatures *c = find_customer(&features, atoi(f[0]));
        if (c == NULL) {
            printf("고객을 찾을 수 없음: %s\n", f[0]);
            return 1;
        }
        y[rows] = atoi(f[1]);
        ref_prob[rows] = atof(f[2]);
        my_prob[rows] = logistic_predict(&model, c);
        rows++;
    }
    fclose(fp);
    if (rows != n_test) {
        printf("시험 고객 수가 다름: %zu / %zu\n", rows, n_test);
        return 1;
    }
    printf("시험 고객 %zu명\n\n", rows);

    int failures = 0;

    /* 1. 유지 고객의 예측창 매출 실측 */
    printf("[시험] 유지 고객의 예측창 매출 (허용 오차 %.0e)\n", PROFIT_TOLERANCE);
    RetainedRevenue retained;
    if (!profit_measure_retained(&tx, &labels, cutoff, label_end, &retained)) {
        return 1;
    }
    failures += check_count("count", (long long)retained.count, EXPECTED_RETAINED_COUNT);
    failures += check_value("median", retained.median, EXPECTED_RETAINED_MEDIAN, PROFIT_TOLERANCE);
    failures += check_value("mean", retained.mean, EXPECTED_RETAINED_MEAN, PROFIT_TOLERANCE);

    /* 2. 쿠폰 1장의 이익 */
    printf("[시험] 쿠폰 1장의 이익\n");
    ProfitAssumptions assumptions = {MARGIN, COUPON_COST, RESPONSE};
    ProfitValues values = profit_values(&assumptions, retained.median);
    failures += check_value("value_tp", values.value_tp, EXPECTED_VALUE_TP, PROFIT_TOLERANCE);
    failures += check_value("value_fp", values.value_fp, EXPECTED_VALUE_FP, PROFIT_TOLERANCE);

    /* 3. 정답 확률로 이익 곡선을 구해 파이썬 값과 비교 */
    printf("[시험] 정답 확률로 이익 곡선 (임계값 0.05 ~ 0.95)\n");
    ProfitPoint curve[PROFIT_THRESHOLDS];
    profit_curve(y, ref_prob, n_test, &values, curve);
    print_curve(curve);

    int best = profit_best_index(curve);
    long long sum_sent = 0;
    double sum_profit = 0.0;
    for (int k = 0; k < PROFIT_THRESHOLDS; k++) {
        sum_sent += curve[k].sent;
        sum_profit += curve[k].profit;
    }
    failures += check_value("best_thr", curve[best].threshold, EXPECTED_BEST_THRESHOLD, PROFIT_TOLERANCE);
    failures += check_value("best_profit", curve[best].profit, EXPECTED_BEST_PROFIT, PROFIT_TOLERANCE);
    failures += check_count("sent@0.30", curve[5].sent, 1056);
    failures += check_count("tp@0.30", curve[5].tp, 643);
    failures += check_count("fp@0.30", curve[5].fp, 413);
    failures += check_value("profit@0.50", curve[9].profit, 6235.709300, PROFIT_TOLERANCE);
    failures += check_count("sent@0.50", curve[9].sent, 776);
    failures += check_count("sum_sent", sum_sent, EXPECTED_SUM_SENT);
    failures += check_value("sum_profit", sum_profit, EXPECTED_SUM_PROFIT, PROFIT_TOLERANCE);

    /* 4. C 가 학습한 모델의 확률로 같은 계산 (정답과 조금 다를 수 있음) */
    printf("[확인] C 모델의 확률로 이익 곡선\n");
    profit_curve(y, my_prob, n_test, &values, curve);
    print_curve(curve);

    free(y);
    free(ref_prob);
    free(my_prob);
    free(train);
    free(test);
    features_free(&features);
    labels_free(&labels);
    returns_free(&returns);
    transactions_free(&tx);

    if (failures == 0) {
        printf("\n모든 시험 통과\n");
        return 0;
    }
    printf("\n실패 %d건\n", failures);
    return 1;
}
