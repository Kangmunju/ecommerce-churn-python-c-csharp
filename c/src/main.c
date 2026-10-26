/* 이탈 예측 파이프라인 (C 구현) */
/* 4단계 : 거래를 메모리에 올리기 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "datetime.h"
#include "transactions.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

#define DEFAULT_PATH "data/processed/export_for_c/clean_transactions.csv"

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

/* 글자 결과가 기대와 같은지 확인하고 출력함 */
static void check_text(const char *name, const char *got, const char *expected) {
    if (strcmp(got, expected) == 0) {
        printf("  OK   %-40s %s\n", name, got);
    } else {
        printf("  실패 %-40s 결과=%s 기대=%s\n", name, got, expected);
        failures++;
    }
}



int main(int argc, char *argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    const char *path = DEFAULT_PATH;
    if (argc > 1) {
        path = argv[1];
    }

    printf("Transaction 한 건의 크기: %zu 바이트\n", sizeof(Transaction));

    TransactionTable table;
    clock_t start = clock();
    if (!transactions_load(path, &table)) {
        return 1;
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("읽은 거래: %zu건, 확보한 칸: %zu칸\n", table.count, table.capacity);
    printf("실제 쓰는 메모리: %.1f MB, 확보한 메모리: %.1f MB\n",
           table.count * sizeof(Transaction) / 1048576.0,
           table.capacity * sizeof(Transaction) / 1048576.0);
    printf("걸린 시간: %.2f초\n", seconds);

    /* 모든 거래를 한 번 훑어서 합계를 구함 */
    long long sum_customer = 0, sum_quantity = 0, sum_timestamp = 0;
    double revenue = 0.0;
    for (size_t i = 0; i < table.count; i++) {
        const Transaction *t = &table.items[i];
        sum_customer += t->customer_id;
        sum_quantity += t->quantity;
        sum_timestamp += t->timestamp;
        revenue += t->quantity * t->price;
    }

    printf("[시험] 파이썬 값과 비교\n");
    check("거래 수", (long long)table.count, 776577);
    check("고객 번호 합계", sum_customer, 11902294299LL);
    check("수량 합계", sum_quantity, 10498435);
    check("초 합계 (③의 값과 같아야 함)", sum_timestamp, 1004902583108760LL);
    check("매출 합계(센트)", (long long)llround(revenue * 100), 1706803087LL);

    const Transaction *first = &table.items[0];
    const Transaction *last = &table.items[table.count - 1];
    char when[32];

    dt_format(first->timestamp, when, sizeof(when));
    check("첫 행 고객", first->customer_id, 13085);
    check_text("첫 행 송장", first->invoice, "489434");
    check_text("첫 행 상품", first->stock_code, "85048");
    check_text("첫 행 일시", when, "2009-12-01 07:45:00");
    check_text("첫 행 국가", first->country, "United Kingdom");

    dt_format(last->timestamp, when, sizeof(when));
    check("마지막 행 고객", last->customer_id, 12680);
    check_text("마지막 행 송장", last->invoice, "581587");
    check_text("마지막 행 일시", when, "2011-12-09 12:50:00");
    check_text("마지막 행 국가", last->country, "France");

    transactions_free(&table);
    check("메모리를 돌려준 뒤 칸 수", (long long)table.capacity, 0);

    if (failures == 0) {
        printf("\n모든 시험 통과\n");
        return 0;
    }
    printf("\n실패 %d건\n", failures);
    return 1;
}
