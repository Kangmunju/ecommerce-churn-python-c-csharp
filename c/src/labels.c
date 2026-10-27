#include "labels.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "datetime.h"


#define NEW_CUSTOMER_DAYS 30




/* bsearch 용: 고객 번호로 라벨을 찾음 */
static int compare_customer_id(const void *key, const void *element) {
    int id = *(const int *)key;
    const CustomerLabel *label = (const CustomerLabel *)element;

    if (id < label->customer_id) {
        return -1;
    }
    if (id > label->customer_id) {
        return 1;
    }
    return 0;
}


int labels_build(const TransactionTable *tx, long long cutoff, long long label_end, LabelTable *table) {
    table->items = NULL;
    table->count = 0;

    size_t n_obs = 0;
    const Transaction **rows = transactions_sorted_observation(tx, cutoff, &n_obs);
    if (rows == NULL) {
        printf("메모리가 부족함: 관측창 거래 %zu건을 정렬하지 못함\n", n_obs);
        return 0;
    }

    /* 1회 */
    /* 고객이 몇 명인지 셈 */
    /* 정렬되어 있어서 번호가 바뀔 때마다 새 고객을 나타냄 */
    size_t n_customers = 0;
    for (size_t i = 0 ; i < n_obs ; i++) {
        if (i == 0 || rows[i]->customer_id != rows[i - 1]->customer_id) {
            n_customers++;
        }
    }

    table->items = malloc(n_customers * sizeof(CustomerLabel));
    if (table->items == NULL) {
        printf("메모리가 부족함: 고객 %zu명의 라벨을 담지 못함\n", n_customers);
        free(rows);
        return 0;
    }
    
    /* 2회 */
    /* 고객마다 처음/마지막 시각과 서로 다른 송장 수를 구함 */
    /* 송장 순으로 정렬되어 있어서 바뀔 때마다 주문이 하나 늘어남 */
    long long new_limit = cutoff - NEW_CUSTOMER_DAYS * SECONDS_PER_DAY;
    for (size_t i = 0 ; i < n_obs ; i++) {
        const Transaction *t = rows[i];
        int is_first_row = (i == 0 || t->customer_id != rows[i - 1]->customer_id);

        if (is_first_row) {
            CustomerLabel *label = &table->items[table->count];
            label->customer_id = t->customer_id;
            label->first_purchase = t->timestamp;
            label->last_purchase = t->timestamp;
            label->n_orders_obs = 1;
            label->churn = 1;
            label->is_new = 0;
            table->count++;
        } else {
            CustomerLabel *label = &table->items[table->count -1];
            if (t->timestamp < label->first_purchase) {
                label->first_purchase = t->timestamp;
            }
            if (t->timestamp > label->last_purchase) {
                label->last_purchase = t->timestamp;
            }
            if (strcmp(t->invoice, rows[i - 1]->invoice) != 0) {
                label->n_orders_obs++;
            }
        }
    }
    free(rows);

    for(size_t i = 0 ; i < table->count ; i++) {
        if (table->items[i].first_purchase >= new_limit) {
            table->items[i].is_new = 1;
        }
    }

    /* 예측창에 구매한 고객은 이탈이 아님 */
    /* 관측창에 없던 고객은 찾지 못하고 넘어감 */
    for (size_t i = 0 ; i < tx->count ; i++) {
        const Transaction *t = &tx->items[i];
        if (t->timestamp < cutoff || t->timestamp >= label_end) {
            continue;
        }
        CustomerLabel *found = bsearch(&t->customer_id, table->items, table->count, sizeof(CustomerLabel), compare_customer_id);
        if (found != NULL) {
            found->churn = 0;
        }
    }
    return 1;
}


void labels_free(LabelTable *table) {
    free(table->items);
    table->items = NULL;
    table->count = 0;
}

/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 해시맵 없이 고객별로 묶는 방법 (이 컴퓨터에서 직접 비교함) */
/*    C 에는 딕셔너리가 없어서 거래를 (고객, 송장) 순으로 정렬하면 같은 고객이 붙어서 한 번 훑으면 끝남 */
/*    예측창 거래마다 고객 배열을 처음부터 훑어 찾으면 비교 7억 1,800만 회, 0.76초 */
/*    정렬 + bsearch 방식은 라벨 만들기 전체가 0.098초 */
/*    이 데이터에서는 단순한 방법도 견딜 만하지만 고객이 늘면 비교 횟수가 그만큼 늘어남 */
/* */
/* 2. 송장 순으로 정렬한 이유 */
/*    거래는 시간순인데 같은 송장이 떨어져 나오는 경우가 있음 (송장이 바뀐 횟수 37,026번, 실제 송장 36,594개) */
/*    시간순으로 훑으며 송장이 바뀔 때 +1 하면 주문 수가 틀림 */
/*    (고객, 송장) 순으로 정렬하면 같은 송장이 붙어서 바뀔 때 +1 이 정확함 */
/*    대신 처음/마지막 시각은 정렬 순서로 알 수 없어서 min/max 로 따로 구함 */
/* */
/* 3. 거래가 아니라 거래의 주소를 정렬한 이유 */
/*    Transaction 은 64바이트라 통째로 정렬하면 옮기는 양이 많고 메모리가 더 듦 */
/*    주소(8바이트)만 모아 정렬하고 원본 거래 표는 그대로 둠 */
/*    그래서 비교 함수가 받는 것은 거래 주소를 담은 칸의 주소임 (const Transaction *const *) */
/* ==================================================================================================== */