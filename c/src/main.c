/* 이탈 예측 파이프라인 (C 구현) */
/* 5단계 : 이탈 라벨 만들기 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csv.h"
#include "datetime.h"
#include "labels.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

#define TX_PATH "data/processed/export_for_c/clean_transactions.csv"
#define REF_PATH "data/processed/export_for_c/reference_features.csv"

static int failures = 0;

/* 정수 결과가 기대와 같은지 확인하고 출력함 */
static void check(const char *name, long long got, long long expected) {
    if (got == expected) {
        printf("  OK   %-40s %lld\n", name, got);
    } else {
        printf("  실패 %-40s 결과=%lld 기대=%lld\n", name, got, expected);
        failures++;
    }
}

int main(void) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    TransactionTable tx;
    if (!transactions_load(TX_PATH, &tx)) {
        return 1;
    }

    long long cutoff, label_end;
    dt_parse("2011-06-10 00:00:00", &cutoff);
    dt_parse("2011-12-10 00:00:00", &label_end);

    LabelTable labels;
    clock_t start = clock();
    if (!labels_build(&tx, cutoff, label_end, &labels)) {
        return 1;
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("CustomerLabel 한 명의 크기: %zu 바이트, 걸린 시간 %.3f초\n",
           sizeof(CustomerLabel), seconds);

    long long churn = 0, is_new = 0, orders = 0;
    for (size_t i = 0; i < labels.count; i++) {
        churn += labels.items[i].churn;
        is_new += labels.items[i].is_new;
        orders += labels.items[i].n_orders_obs;
    }

    printf("[시험] 파이썬 값과 비교\n");
    check("대상 고객", (long long)labels.count, 4944);
    check("이탈 고객", churn, 2372);
    check("신규 고객", is_new, 105);
    check("주문 수 합계(frequency 합)", orders, 26033);

    /* reference_features.csv 와 고객별로 비교 */
    FILE *fp = fopen(REF_PATH, "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: %s\n", REF_PATH);
        return 1;
    }
    char line[2048];
    char *f[CSV_MAX_FIELDS];
    csv_read_line(fp, line, sizeof(line));

    size_t row = 0;
    long long mismatch = 0;
    while (csv_read_line(fp, line, sizeof(line)) == 1) {
        int n = csv_split(line, f, CSV_MAX_FIELDS);
        int id = atoi(f[0]);
        int ref_freq = (int)atof(f[2]);
        int ref_churn = atoi(f[n - 2]);
        int ref_new = atoi(f[n - 1]);

        if (row >= labels.count || labels.items[row].customer_id != id ||
            labels.items[row].churn != ref_churn ||
            labels.items[row].is_new != ref_new ||
            labels.items[row].n_orders_obs != ref_freq) {
            mismatch++;
        }
        row++;
    }
    fclose(fp);
    check("정답 CSV 행 수", (long long)row, 4944);
    check("고객별 불일치(번호·이탈·신규·주문수)", mismatch, 0);

    char first[32], last[32];
    dt_format(labels.items[0].first_purchase, first, sizeof(first));
    dt_format(labels.items[0].last_purchase, last, sizeof(last));
    printf("첫 고객 %d: %s ~ %s\n", labels.items[0].customer_id, first, last);

    labels_free(&labels);
    transactions_free(&tx);

    if (failures == 0) {
        printf("\n모든 시험 통과\n");
        return 0;
    }
    printf("\n실패 %d건\n", failures);
    return 1;
}
