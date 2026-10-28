/* 이탈 예측 파이프라인 (C 구현) */
/* 7 단계 : 표준화 + 로지스틱 회귀 */


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
#include "returns.h"
#include "transactions.h"


/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif


#define EXPORT_DIR "data/processed/export_for_c/"

/* 끝까지 수렴시킨 정답과의 허용 오차 */
/* 절대 오차 */
#define TIGHT_TOLERANCE 1e-5

/* 정답 CSV는 sklearn 이 일찍 멈춘 값이라 이 정도 차이를 허용함 */
/* 하단의 설계 메모를 참고할 것 */
#define CSV_COEF_TOLERANCE 2e-2
#define CSV_PROB_TOLERANCE 1e-2


/* tools/expected_values.py 의 [로지스틱 회귀 (tol=1e-12)] 출력 */
static const double EXPECTED_INTERCEPT = -0.3340369420;
static const double EXPECTED_COEF[FEATURE_COUNT] = {
    +0.4506646029, -0.0889310089, +0.0520370798, -0.1251122816, -0.0736773152,
    +0.1633841582, -0.1691433271, +0.0140261778, -0.3081052258, +0.2612149893,
    +0.0732026197, +0.1435904920, +0.0613287830, -1.4657474484, +0.1853803167,
    +0.3032913514, -0.4637288594, +0.1934765540, -0.1370722240, -0.0464534615};


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
    printf("학습 %zu명, 시험 %zu명\n", n_train, n_test);

    LogisticModel model;
    clock_t start = clock();
    if (!logistic_fit(&features, train, n_train, 1.0, &model)) {
        return 1;
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("학습 완료: 뉴턴 반복 %d회, 걸린 시간 %.3f초\n\n", model.iterations, seconds);

    int failures = 0;

    /* 1. 표준화 평균, 표준편차를 reference_coefficients.csv 와 비교 */
    FILE *fp = fopen(EXPORT_DIR "reference_coefficients.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_coefficients.csv\n");
        return 1;
    }
    char line[256];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));
    double max_mean = 0.0, max_scale = 0.0, max_csv_coef = 0.0, max_tight_coef = 0.0;
    for (int k = 0; k < FEATURE_COUNT; k++) {
        csv_read_line(fp, line, sizeof(line));
        csv_split(line, f, CSV_MAX_FIELDS);
        double e_mean = fabs(model.mean[k] - atof(f[1])) / fmax(1.0, fabs(atof(f[1])));
        double e_scale = fabs(model.scale[k] - atof(f[2])) / fmax(1.0, fabs(atof(f[2])));
        double e_csv = fabs(model.coef[k] - atof(f[3]));
        double e_tight = fabs(model.coef[k] - EXPECTED_COEF[k]);
        if (e_mean > max_mean) max_mean = e_mean;
        if (e_scale > max_scale) max_scale = e_scale;
        if (e_csv > max_csv_coef) max_csv_coef = e_csv;
        if (e_tight > max_tight_coef) max_tight_coef = e_tight;
    }
    fclose(fp);

    printf("[시험] 표준화 (reference_coefficients.csv, 허용 상대 오차 1e-9)\n");
    if (max_mean <= 1e-9 && max_scale <= 1e-9) {
        printf("  OK   평균 %.2e, 표준편차 %.2e\n", max_mean, max_scale);
    } else {
        printf("  실패 평균 %.2e, 표준편차 %.2e\n", max_mean, max_scale);
        failures++;
    }

    /* 2. 계수, 절편을 끝까지 수렴시킨 정답과 비교 */
    printf("[시험] 계수와 절편 (tol=1e-12 정답, 허용 오차 %.0e)\n", TIGHT_TOLERANCE);
    double e_intercept = fabs(model.intercept - EXPECTED_INTERCEPT);
    if (max_tight_coef <= TIGHT_TOLERANCE && e_intercept <= TIGHT_TOLERANCE) {
        printf("  OK   계수 최대 오차 %.2e, 절편 오차 %.2e\n", max_tight_coef, e_intercept);
    } else {
        printf("  실패 계수 최대 오차 %.2e, 절편 오차 %.2e\n", max_tight_coef, e_intercept);
        failures++;
    }

    /* 3. 정답 CSV 의 계수와도 비교 (정답 쪽이 덜 수렴해 차이가 큼) */
    printf("[시험] 계수 (reference_coefficients.csv, 허용 오차 %.0e)\n", CSV_COEF_TOLERANCE);
    if (max_csv_coef <= CSV_COEF_TOLERANCE) {
        printf("  OK   최대 오차 %.2e\n", max_csv_coef);
    } else {
        printf("  실패 최대 오차 %.2e\n", max_csv_coef);
        failures++;
    }

    /* 4. 시험 고객의 이탈 확률을 reference_test_pred.csv 와 비교 */
    fp = fopen(EXPORT_DIR "reference_test_pred.csv", "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: reference_test_pred.csv\n");
        return 1;
    }
    csv_read_line(fp, line, sizeof(line));
    double max_prob = 0.0;
    size_t rows = 0;
    int missing = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, f, CSV_MAX_FIELDS);
        const CustomerFeatures *c = find_customer(&features, atoi(f[0]));
        if (c == NULL) {
            missing++;
            continue;
        }
        double error = fabs(logistic_predict(&model, c) - atof(f[2]));
        if (error > max_prob) {
            max_prob = error;
        }
        rows++;
    }
    fclose(fp);

    printf("[시험] 시험 고객 확률 (reference_test_pred.csv, 허용 오차 %.0e)\n", CSV_PROB_TOLERANCE);
    if (missing == 0 && rows == n_test && max_prob <= CSV_PROB_TOLERANCE) {
        printf("  OK   %zu명, 최대 오차 %.2e\n", rows, max_prob);
    } else {
        printf("  실패 %zu명(기대 %zu), 없는 고객 %d, 최대 오차 %.2e\n", rows, n_test, missing, max_prob);
        failures++;
    }

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