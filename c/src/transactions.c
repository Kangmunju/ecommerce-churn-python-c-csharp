#include "transactions.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "csv.h"
#include "datetime.h"


#define LINE_SIZE 1024
#define INITIAL_CAPACITY 1024
#define EXPECTED_FIELDS 7


/* src 를 size byte짜리 칸인 dest에 복사함 */
/* 너무 길어서 들어가지 않는 경우 0 리턴하도록 함 */
static int copy_field(char *dest, size_t size, const char *src) {
    if (strlen(src) >= size) {
        return 0;
    }
    strcpy(dest, src);
    return 1;
}


/* 한 줄의 필드 7개를 Transaction 하나로 교체 */
/* 실패하면 0을 리턴하도록 함 */
static int parse_row(char *fields[], Transaction *t) {
    t->customer_id = (int)strtol(fields[0], NULL, 10);
    t->quantity = (int)strtol(fields[4], NULL, 10);
    t->price = strtod(fields[5],NULL);

    if(!dt_parse(fields[3], &t->timestamp)) {
        return 0;
    }
    if (!copy_field(t->invoice, sizeof(t->invoice), fields[1])) {
        return 0;
    }
    if (!copy_field(t->stock_code, sizeof(t->stock_code), fields[2])) {
        return 0;
    }
    if (!copy_field(t->country, sizeof(t->country), fields[6])) {
        return 0;
    }
    return 1;
}


/* 칸이 모자라면 두 배로 늘림 */
/* 메모리가 없으면 0을 리턴함 */
static int grow(TransactionTable *table) {
    size_t new_capacity = table->capacity * 2;
    if(new_capacity == 0) {
        new_capacity = INITIAL_CAPACITY;
    }

    /* realloc이 실패하면 NULL을 받게 됨 */
    /* table->items 에 바로 받으면 실패 시 items가 NULL로 덮여 이전 메모리를 잃을 가능성 있음 */
    /* bigger라는 임시 변수로 성공했을 때만 옮기도록 작성함 */
    Transaction *bigger = 
      realloc(table->items, new_capacity * sizeof(Transaction));
    if (bigger == NULL) {
        return 0;
    }

    table->items = bigger;
    table->capacity = new_capacity;
    return 1;
}


int transactions_load(const char *path, TransactionTable *table) {
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

    /* 첫 줄은 열 이름이므로 읽고 버림 */
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
        transactions_free(table);
        return 0;
    }
    return 1;
}


void transactions_free(TransactionTable *table) {
    free(table->items);
    table->items = NULL;
    table->count = 0;
    table->capacity = 0;
}

/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 구조체 필드를 큰 것부터 적은 이유 (이 컴퓨터에서 직접 확인함) */
/*    8바이트 값(long long, double)은 8의 배수 위치에 놓여서 중간에 빈 칸(패딩)이 생김 */
/*    자연스러운 순서(customer_id, invoice, ...)는 72바이트, 큰 것부터 적으면 64바이트 */
/*    77만 건 기준 53.3MB -> 47.4MB */
/* */
/* 2. 칸을 두 배씩 늘리는 이유 */
/*    파일을 읽기 전에는 거래 수를 모르므로 1,024칸으로 시작해 꽉 차면 두 배로 늘림 */
/*    한 칸씩 늘리면 매번 전체 복사가 일어나서 느림 */
/*    77만 건에서 최종 칸 수는 1,048,576 (실제 쓰는 47.4MB, 확보한 64.0MB) */
/*    늘리는 순간 옛 배열과 새 배열이 잠시 같이 있어서 최대 메모리는 약 74.6MB */
/* */
/* 3. 문자열 칸 크기와 복사 */
/*    송장 6글자, 상품 코드 7글자, 국가 20글자 이하 (데이터 점검 결과) + 끝 표시 1칸 */
/*    strcpy 는 칸이 작아도 옆 메모리를 덮어쓰므로 copy_field 가 길이를 먼저 확인함 */
/* */
/* 4. 메모리 누수 확인 */
/*    적재와 해제를 20번 더 반복해도 메모리가 10.4MB 로 돌아와서 누수가 없음 */
/* ==================================================================================================== */