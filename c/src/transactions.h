#ifndef TRANSACTIONS_H
#define TRANSACTIONS_H


#include <stddef.h>


/* 거래 한 건 */
/* clean_transactions.csv 의 한 행 */
/* 큰 필드부터 적어서 빈 칸(패딩) 없이 64바이트가 되도록 작성함 */
typedef struct {

  /* 1970-01-01 00:00:00 부터 지난 초 */
    long long timestamp;
    double price;
    int customer_id;
    int quantity;

    /* 최대 6글자 + 끝 표시 */
    char invoice[8];

    /* 최대 7글자 + 끝 표시 */
    char stock_code[8];

    /* 최대 20글자 + 끝 표시 */
    char country[24];
} Transaction;


/* 거래 전체를 담는 표 */
/* 거래 배열 + 현재 몇 건 + 몇 칸 확보 -> 한 묶음으로 생성함 */
typedef struct 
{
    /* 거래 배열의 시작 주소 */
    Transaction *items;

    /* 담긴 거래 수 */
    size_t count;

    /* 확보해 둔 칸 수 */
    size_t capacity;
} TransactionTable;


/* CSV를 읽어 table에 채움 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패하는 경우 이유를 화면에 출력함 */
int transactions_load(const char *path, TransactionTable *table);


/* table 이 쓰던 메모리를 돌려줌 */
void transactions_free(TransactionTable * table);


/* 관측창(timestamp < cutoff)의 거래 주소를 모아 (고객, 송장) 순으로 정렬해서 돌려줌 */
/* 개수는 n_obs 로 돌려주고, 메모리가 없으면 NULL */
/* 다 쓰고 나면 free 로 돌려줘야 함 (거래 자체는 table 이 갖고 있음) */
const Transaction **transactions_sorted_observation(const TransactionTable *table, long long cutoff, size_t *n_obs);


#endif

