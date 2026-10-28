/* 이탈 예측 파이프라인 (C 구현) */
/* 8 단계 : 평가 지표 */

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
#include "returns.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif


#define EXPORT_DIR "data/processed/export_for_c/"

/* 같은 확률로 계산한 값끼리 비교하므로 매우 엄격하게 봄 (상대 오차) */
#define METRIC_TOLERANCE 1e-9
/* C 가 직접 학습한 모델의 확률은 정답과 최대 8e-3 달라서 지표도 이 정도는 달라질 수 있음 */
#define MODEL_TOLERANCE 2e-3

/* tools/expected_values.py 의 [reference_test_pred.csv, 임계값 0.5] 출력 */
#define EXPECTED_AUC 0.781833993130
#define EXPECTED_PR_AUC 0.750203587894
#define EXPECTED_ACCURACY 0.712938005391
#define EXPECTED_PRECISION 0.684278350515
#define EXPECTED_RECALL 0.745786516854
#define EXPECTED_F1 0.713709677419
#define EXPECTED_BRIER 0.190716936782
/* 확률을 전부 0.5 로 둔 경우 */
#define EXPECTED_FLAT_AUC 0.500000000000
#define EXPECTED_FLAT_PR_AUC 0.479784366577


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
        printf("  OK   %-10s %.6f (오차 %.2e)\n", name, actual, error);
        return 0;
    }
    printf("  실패 %-10s %.6f, 기대 %.6f (오차 %.2e)\n", name, actual, expected, error);
    return 1;
}


/* 혼동행렬이 기대하는 개수와 같은지 확인함 */
/* 돌려주는 값: 실패 1, 통과 0 */
static int check_confusion(const ConfusionMatrix *cm, long long tn, long long fp, long long fn,
                           long long tp) {
    if (cm->tn == tn && cm->fp == fp && cm->fn == fn && cm->tp == tp) {
        printf("  OK   혼동행렬     TN %lld, FP %lld, FN %lld, TP %lld\n", cm->tn, cm->fp, cm->fn, cm->tp);
        return 0;
    }
    printf("  실패 혼동행렬     TN %lld, FP %lld, FN %lld, TP %lld (기대 %lld %lld %lld %lld)\n",
           cm->tn, cm->fp, cm->fn, cm->tp, tn, fp, fn, tp);
    return 1;
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
    double *flat_prob = malloc(n_test * sizeof(double));
    if (y == NULL || ref_prob == NULL || my_prob == NULL || flat_prob == NULL) {
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
        flat_prob[rows] = 0.5;
        rows++;
    }
    fclose(fp);
    if (rows != n_test) {
        printf("시험 고객 수가 다름: %zu / %zu\n", rows, n_test);
        return 1;
    }
    printf("시험 고객 %zu명\n\n", rows);

    int failures = 0;
    Metrics m;

    /* 1. 파이썬이 계산한 확률로 지표를 구해 sklearn 값과 비교 (지표 계산 코드만 시험함) */
    printf("[시험] 정답 확률로 지표 계산 (임계값 0.5, 허용 오차 %.0e)\n", METRIC_TOLERANCE);
    if (!metrics_evaluate(y, ref_prob, n_test, 0.5, &m)) {
        return 1;
    }
    failures += check_value("AUC", m.auc, EXPECTED_AUC, METRIC_TOLERANCE);
    failures += check_value("PR-AUC", m.pr_auc, EXPECTED_PR_AUC, METRIC_TOLERANCE);
    failures += check_value("accuracy", m.accuracy, EXPECTED_ACCURACY, METRIC_TOLERANCE);
    failures += check_value("precision", m.precision, EXPECTED_PRECISION, METRIC_TOLERANCE);
    failures += check_value("recall", m.recall, EXPECTED_RECALL, METRIC_TOLERANCE);
    failures += check_value("F1", m.f1, EXPECTED_F1, METRIC_TOLERANCE);
    failures += check_value("Brier", m.brier, EXPECTED_BRIER, METRIC_TOLERANCE);
    failures += check_confusion(&m.cm, 527, 245, 181, 531);

    /* 2. 확률이 전부 같은 경우 (동점 처리) */
    printf("[시험] 확률이 전부 0.5 인 경우\n");
    if (!metrics_evaluate(y, flat_prob, n_test, 0.5, &m)) {
        return 1;
    }
    failures += check_value("AUC", m.auc, EXPECTED_FLAT_AUC, METRIC_TOLERANCE);
    failures += check_value("PR-AUC", m.pr_auc, EXPECTED_FLAT_PR_AUC, METRIC_TOLERANCE);

    /* 3. C 가 학습한 모델의 확률로 지표를 구해 정답과 비교 */
    printf("[시험] C 모델의 확률로 지표 계산 (허용 오차 %.0e)\n", MODEL_TOLERANCE);
    if (!metrics_evaluate(y, my_prob, n_test, 0.5, &m)) {
        return 1;
    }
    failures += check_value("AUC", m.auc, EXPECTED_AUC, MODEL_TOLERANCE);
    failures += check_value("PR-AUC", m.pr_auc, EXPECTED_PR_AUC, MODEL_TOLERANCE);
    failures += check_value("accuracy", m.accuracy, EXPECTED_ACCURACY, MODEL_TOLERANCE);
    failures += check_value("precision", m.precision, EXPECTED_PRECISION, MODEL_TOLERANCE);
    failures += check_value("recall", m.recall, EXPECTED_RECALL, MODEL_TOLERANCE);
    failures += check_value("F1", m.f1, EXPECTED_F1, MODEL_TOLERANCE);
    failures += check_value("Brier", m.brier, EXPECTED_BRIER, MODEL_TOLERANCE);
    printf("  혼동행렬 TN %lld, FP %lld, FN %lld, TP %lld\n", m.cm.tn, m.cm.fp, m.cm.fn, m.cm.tp);

    free(y);
    free(ref_prob);
    free(my_prob);
    free(flat_prob);
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
