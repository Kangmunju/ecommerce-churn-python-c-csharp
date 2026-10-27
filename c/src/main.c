/* 이탈 예측 파이프라인 (C 구현) */
/* 6-2단계 : 피처 20개 만들기 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv.h"
#include "datetime.h"
#include "features.h"
#include "labels.h"
#include "returns.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

#define TX_PATH "data/processed/export_for_c/clean_transactions.csv"
#define RETURNS_PATH "data/processed/export_for_c/returns.csv"
#define REF_PATH "data/processed/export_for_c/reference_features.csv"

/* 정답과의 차이가 이 값(상대 오차)보다 작으면 같다고 봄 */
#define TOLERANCE 1e-9

int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    TransactionTable tx;
    if (!transactions_load(TX_PATH, &tx)) {
        return 1;
    }
    ReturnTable returns;
    if (!returns_load(RETURNS_PATH, &returns)) {
        return 1;
    }
    printf("거래 %zu건, 반품 %zu건\n", tx.count, returns.count);

    long long cutoff, label_end;
    dt_parse("2011-06-10 00:00:00", &cutoff);
    dt_parse("2011-12-10 00:00:00", &label_end);

    LabelTable labels;
    if (!labels_build(&tx, cutoff, label_end, &labels)) {
        return 1;
    }

    FeatureTable features;
    clock_t start = clock();
    if (!features_build(&tx, &returns, &labels, cutoff, &features)) {
        return 1;
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("피처 %zu명 x %d개, 걸린 시간 %.3f초\n", features.count, FEATURE_COUNT, seconds);

    /* reference_features.csv 와 고객별, 피처별로 비교 */
    FILE *fp = fopen(REF_PATH, "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: %s\n", REF_PATH);
        return 1;
    }
    char line[2048];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));

    double max_error[FEATURE_COUNT] = {0.0};
    long long wrong_label = 0;
    size_t row = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        csv_split(line, f, CSV_MAX_FIELDS);
        if (row >= features.count || features.items[row].customer_id != atoi(f[0])) {
            printf("고객 순서가 다름: %zu행\n", row);
            return 1;
        }

        const CustomerFeatures *c = &features.items[row];
        for (int k = 0; k < FEATURE_COUNT; k++) {
            double expected = atof(f[k + 1]);
            double error = fabs(c->x[k] - expected) / fmax(1.0, fabs(expected));
            if (error > max_error[k]) {
                max_error[k] = error;
            }
        }
        if (c->churn != atoi(f[FEATURE_COUNT + 1]) || c->is_new != atoi(f[FEATURE_COUNT + 2])) {
            wrong_label++;
        }
        row++;
    }
    fclose(fp);

    /* 기대값은 숫자를 적지 않고 파이썬 정답지가 내보낸 reference_features.csv 를 읽어 비교함 */
    /* 이 파일은 tools/export_for_c.py 가 파이썬 피처 결과(features.parquet)에서 만듦 */
    printf("[시험] 정답 CSV %zu행과 비교 (허용 상대 오차 %.0e)\n", row, TOLERANCE);
    int failures = 0;
    for (int k = 0; k < FEATURE_COUNT; k++) {
        if (max_error[k] <= TOLERANCE) {
            printf("  OK   %-16s 최대 오차 %.2e\n", FEATURE_NAMES[k], max_error[k]);
        } else {
            printf("  실패 %-16s 최대 오차 %.2e\n", FEATURE_NAMES[k], max_error[k]);
            failures++;
        }
    }
    if (wrong_label != 0 || row != features.count) {
        printf("  실패 라벨 불일치 %lld건, 행 수 %zu / %zu\n", wrong_label, row, features.count);
        failures++;
    }

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
