#include "returns.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "csv.h"
#include "datetime.h"


#define LINE_SIZE 1024
#define INITIAL_CAPACITY 1024
#define EXPECTED_FIELDS 3



/* 한 줄의 필드 3개를 Return 하나로 교체함 */
/* 필드 순서: fields[0] = 고객, [1] = 송장, [2] = 일시 */
/* 실패하는 경우 0 리턴 */
static int parse_row(char *fields[], Return *r) {
    r->customer_id = (int)strtol(fields[0], NULL, 10);

    if (!dt_parse(fields[2], &r->timestamp)) {
        return 0;
    }
    if (strlen(fields[1]) >= sizeof(r->invoice)) {
        return 0;
    }
    strcpy(r->invoice, fields[1]);
    return 1;
}


/* 칸이 모자라면 두 배로 늘림 */
/* 메모리가 없으면 0 리턴 */
static int grow(ReturnTable *table) {
    size_t new_capacity = table->capacity * 2;
    if (new_capacity == 0) {
        new_capacity = INITIAL_CAPACITY;
    }

    Return *bigger = realloc(table->items, new_capacity * sizeof(Return));
    if (bigger == NULL) {
        return 0;
    }

    table->items = bigger;
    table->capacity = new_capacity;
    return 1;
}


int returns_load(const char *path, ReturnTable *table) {
    table->items = NULL;
    table->count = 0;
    table->capacity = 0;

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        printf("파일을 열 수 없음: %s\n", path);
        return 0;
    }

    char line[LINE_SIZE];
    char *fields[CSV_MAX_FIELDS];

    /* 첫 줄은 열 이름이기 때문에 읽고 버리도록 함 */
    csv_read_line(fp, line, sizeof(line));

    int ok = 1;
    int rc;
    long long line_no = 1;
    while ((rc = csv_read_line(fp, line, sizeof(line))) == 1) {
        line_no++;

        int n = csv_split(line, fields, CSV_MAX_FIELDS);
        if (n != EXPECTED_FIELDS) {
            printf("필드 수가 %d 이 아님: %lld행, %d개\n", EXPECTED_FIELDS, line_no, n);
            ok = 0;
            break;
        }

        if (table->count == table->capacity && !grow(table)) {
            printf("메모리가 부족함: %lld행에서 칸을 늘리지 못함\n", line_no);
            ok = 0;
            break;
        }

        if (!parse_row(fields, &table->items[table->count])) {
            printf("해석할 수 없는 행: %lld행\n", line_no);
            ok = 0;
            break;
        }
        table->count++;
    }
    fclose(fp);

    if (rc < 0) {
        printf("줄이 %d 바이트보다 길어서 잘림: %lld행 근처\n", LINE_SIZE, line_no + 1);
        ok = 0;
    }

    /* 실패한 경우 이미 잡은 메모리를 반환하고 마치도록 함 */
    if (!ok) {
        returns_free(table);
        return 0;
    }
    return 1;
}


void returns_free(ReturnTable *table) {
    free(table->items);
    table->items = NULL;
    table->count = 0;
    table->capacity = 0;
}

/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 송장 칸 크기와 구조체 크기 */
/*    반품 송장은 맨 앞에 C 가 붙어 최대 7글자 (C489449) 라서 끝 표시까지 char invoice[8] 이 딱 맞음 */
/*    필드 크기의 합은 20바이트인데 가장 큰 필드(8바이트)의 배수로 맞춰져서 sizeof(Return) 은 24바이트 */
/*    (끝에 빈 칸 4바이트, Transaction 처럼 딱 맞출 방법은 없음, 18,390건이라 약 430KB) */
/* */
/* 2. transactions.c 와 거의 같은 코드인 이유 */
/*    파일 열기, 첫 줄 건너뛰기, 읽기 반복, 칸 두 배 늘리기, 해제가 타입 이름만 다르고 같음 */
/*    C 에는 어떤 타입이든 담는 표(제네릭)가 없어서 타입마다 같은 코드를 다시 써야 함 */
/* ==================================================================================================== */