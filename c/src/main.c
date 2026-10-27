/* 이탈 예측 파이프라인 (C 구현) */
/* 6-1단계 : 반품 전표 읽기 */

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "datetime.h"
#include "returns.h"

/* Windows 에서만 콘솔 출력을 UTF-8 로 바꾸기 위한 헤더 */
#ifdef _WIN32
#include <windows.h>
#endif

#define DEFAULT_PATH "data/processed/export_for_c/returns.csv"

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

    printf("Return 한 건의 크기: %zu 바이트\n", sizeof(Return));

    ReturnTable table;
    clock_t start = clock();
    if (!returns_load(path, &table)) {
        return 1;
    }
    double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("읽은 반품: %zu건, 확보한 칸: %zu칸, 걸린 시간: %.3f초\n",
           table.count, table.capacity, seconds);

    /* 모든 반품을 한 번 훑어서 합계를 구함 */
    long long sum_customer = 0, sum_timestamp = 0, before_cutoff = 0;
    long long cutoff;
    dt_parse("2011-06-10 00:00:00", &cutoff);
    for (size_t i = 0; i < table.count; i++) {
        sum_customer += table.items[i].customer_id;
        sum_timestamp += table.items[i].timestamp;
        if (table.items[i].timestamp < cutoff) {
            before_cutoff++;
        }
    }

    /* 아래 기대값은 파이썬(pandas)으로 같은 returns.csv 를 읽어 미리 구한 값임 */
    /* 행 수, customer_id 합계, 첫 행과 마지막 행, invoice_date < 2011-06-10 인 행 수 */
    /* 초 합계는 각 invoice_date 를 1970-01-01 00:00:00 과의 차이(초)로 바꿔 더한 값 */
    printf("[시험] 파이썬 값과 비교\n");
    check("반품 수", (long long)table.count, 18390);
    check("고객 번호 합계", sum_customer, 275994751LL);
    check("초 합계", sum_timestamp, 23761154531760LL);
    check("2011-06-10 이전 반품 수", before_cutoff, 13309);

    const Return *first = &table.items[0];
    const Return *last = &table.items[table.count - 1];
    char when[32];

    dt_format(first->timestamp, when, sizeof(when));
    check("첫 행 고객", first->customer_id, 16321);
    check_text("첫 행 송장", first->invoice, "C489449");
    check_text("첫 행 일시", when, "2009-12-01 10:33:00");

    dt_format(last->timestamp, when, sizeof(when));
    check("마지막 행 고객", last->customer_id, 17315);
    check_text("마지막 행 송장", last->invoice, "C581569");
    check_text("마지막 행 일시", when, "2011-12-09 11:58:00");

    returns_free(&table);
    check("메모리를 돌려준 뒤 칸 수", (long long)table.capacity, 0);

    if (failures == 0) {
        printf("\n모든 시험 통과\n");
        return 0;
    }
    printf("\n실패 %d건\n", failures);
    return 1;
}
