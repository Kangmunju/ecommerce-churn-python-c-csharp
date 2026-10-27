#ifndef LABELS_H
#define LABELS_H


#include <stddef.h>


#include "transactions.h"


/* 고객 한 명의 이탈 라벨과 관측창 요약 */
/* 큰 필드인 long long 2개 먼저 적어서 패딩 없이 32바이트가 되도록 작성함 */
typedef struct 
{
    
    /* 관측창에서 처음/마지막으로 산 시각 */
    /* 1970-01-01 00:00:00 부터 지난 초 */
    long long first_purchase;
    long long last_purchase;

    int customer_id;

    /* 관측창에서 서로 다른 송장 수 계산 */
    int n_orders_obs;

    /* 1 = 예측창에 구매 없음(이탈) */
    /* 0 = 예측창에 구매 있음 */
    int churn;

    /* 1 = 관측창 마지막 30일 안에 첫 구매 */
    int is_new;
} CustomerLabel;


/* 라벨 전체를 담는 표 */
/* customer_id 오름차순으로 작성 */
typedef struct 
{
    CustomerLabel *items;
    size_t count;
} LabelTable;


/* 관측창: timestamp < cutoff */
/* 예측창: cutoff <= timestamp < label_end */
/* 돌려주는 값이 1이면 성공, 0이면 실패 */
/* 실패시 이유를 화면에 출력함 */
int labels_build(const TransactionTable *tx, long long cutoff, long long label_end, LabelTable *table);

void labels_free(LabelTable *table);


#endif

