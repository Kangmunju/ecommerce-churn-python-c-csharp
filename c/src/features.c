#include "features.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "datetime.h"

#define RECENT_DAYS 90
#define RECENCY_RATIO_MAX 50.0

const char *const FEATURE_NAMES[FEATURE_COUNT] = {
    "recency", "frequency", "monetary", "avg_order_value", "std_order_value",
    "max_order_value", "total_items", "tenure", "active_span",
    "gap_mean", "gap_std", "gap_max", "recency_ratio",
    "orders_90d", "revenue_90d", "share_90d",
    "n_products", "n_returns", "return_rate", "is_uk"};

/* 주문 한 건 = 같은 고객의 같은 송장 */
/* 거래 줄을 송장 단위로 접은 것 */
/* Order는 features.c 안에서만 사용하는 구조체라서 .h가 아닌 .c에 두었음 */
typedef struct {
    /* 송장의 가장 이른 시각 (order_date) */
    long long first_time;
    
    /* 송장의 가장 늦은 시각 */
    long long last_time;

    /* 수량 x 가격의 합 */
    double value;

    /* 수량의 합 */
    double items;

    int customer_id;

    /* 가장 늦은 줄의 국가가 영국이면 1 */
    int last_is_uk;
} Order;


static int compare_orders_by_customer_date(const void *a, const void *b) {
    const Order *x = (const Order *)a;
    const Order *y = (const Order *)b;

    if (x->customer_id < y->customer_id) {
        return -1;
    }
    if (x->customer_id > y->customer_id) {
        return 1;
    }
    if (x->first_time < y->first_time) {
        return -1;
    }
    if (x->first_time > y->first_time) {
        return 1;
    }
    return 0;
}

static int compare_by_customer_stock(const void *a, const void *b) {
    const Transaction *x = *(const Transaction *const *)a;
    const Transaction *y = *(const Transaction *const *)b;

    if (x->customer_id < y->customer_id) {
        return -1;
    }
    if (x->customer_id > y->customer_id) {
        return 1;
    }
    return strcmp(x->stock_code, y->stock_code);
}

static int compare_returns_by_customer_invoice(const void *a, const void *b) {
    const Return *x = *(const Return *const *)a;
    const Return *y = *(const Return *const *)b;

    if (x->customer_id < y->customer_id) {
        return -1;
    }
    if (x->customer_id > y->customer_id) {
        return 1;
    }
    return strcmp(x->invoice, y->invoice);
}

static int compare_feature_customer_id(const void *key, const void *element) {
    int id = *(const int *)key;
    const CustomerFeatures *f = (const CustomerFeatures *)element;

    if (id < f->customer_id) {
        return -1;
    }
    if (id > f->customer_id) {
        return 1;
    }
    return 0;
}



/* 1회 -> 주문이 몇 건인지 셈 */
/* 2회 -> 채움 */
/* 정렬되어 있으므로 (고객, 송장)이 바뀌는 곳이 새주문의 시작을 나타내도록 함 */
/* 줄마다 주문 하나에 누적함 */
static Order *fold_orders(const Transaction **rows, size_t n_obs, size_t *n_orders) {
    size_t count = 0;
    for (size_t i = 0; i < n_obs; i++) {
        if (i == 0 || rows[i]->customer_id != rows[i - 1]->customer_id ||
            strcmp(rows[i]->invoice, rows[i - 1]->invoice) != 0) {
            count++;
        }
    }

    Order *orders = malloc(count * sizeof(Order));
    if (orders == NULL) {
        return NULL;
    }

    size_t k = 0;
    for (size_t i = 0; i < n_obs; i++) {
        const Transaction *t = rows[i];
        int is_uk = (strcmp(t->country, "United Kingdom") == 0);
        int is_first_row = (i == 0 || t->customer_id != rows[i - 1]->customer_id ||
                            strcmp(t->invoice, rows[i - 1]->invoice) != 0);

        if (is_first_row) {
            Order *first = &orders[k];
            first->customer_id = t->customer_id;
            first->first_time = t->timestamp;
            first->last_time = t->timestamp;
            first->value = 0.0;
            first->items = 0.0;
            first->last_is_uk = is_uk;
            k++;
        }

        Order *o = &orders[k - 1];
        o->value += t->quantity * t->price;
        o->items += t->quantity;
        if (t->timestamp < o->first_time) {
            o->first_time = t->timestamp;
        }
        if (t->timestamp > o->last_time) {
            o->last_time = t->timestamp;
            o->last_is_uk = is_uk;
        }
    }
    *n_orders = count;
    return orders;
}


/* 한 고객의 주문 목록(날짜순으로 정렬된 연속 구간)을 받는 함수를 작성함 */
/* orders: 그 배열 중 고객의 첫 주문의 주소 */
/* n: 그 고객의 주문 수*/
/* 고객 한 명의 주문 목록(날짜순)에서 RFM, 주문 간격, 최근 90일, 국가 피처를 채움 */
static void fill_order_features(const Order *orders, size_t n, long long cutoff,
                                double *x) {
    long long recent_start = cutoff - RECENT_DAYS * SECONDS_PER_DAY;

    double sum_value = 0.0, max_value = 0.0, items = 0.0;
    double orders_90d = 0.0, revenue_90d = 0.0;
    long long first_time = orders[0].first_time;
    long long last_time = orders[0].first_time;
    long long latest_row_time = orders[0].last_time;
    int is_uk = orders[0].last_is_uk;

    for (size_t i = 0; i < n; i++) {
        sum_value += orders[i].value;
        items += orders[i].items;
        if (i == 0 || orders[i].value > max_value) {
            max_value = orders[i].value;
        }
        if (orders[i].first_time >= recent_start) {
            orders_90d += 1.0;
            revenue_90d += orders[i].value;
        }
        if (orders[i].first_time > last_time) {
            last_time = orders[i].first_time;
        }
        if (orders[i].last_time >= latest_row_time) {
            latest_row_time = orders[i].last_time;
            is_uk = orders[i].last_is_uk;
        }
    }

    double mean_value = sum_value / (double)n;
    double std_value = 0.0;
    if (n > 1) {
        double ss = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = orders[i].value - mean_value;
            ss += d * d;
        }
        std_value = sqrt(ss / (double)(n - 1));
    }

    double recency = (double)dt_days_between(cutoff, last_time);
    double tenure = (double)dt_days_between(cutoff, first_time);

    /* 주문 간격: 1회 구매 고객은 간격이 없어서 지금까지 기다린 기간(tenure)으로 채움 */
    double gap_mean = tenure, gap_std = 0.0, gap_max = tenure;
    if (n > 1) {
        double gap_sum = 0.0;
        gap_max = 0.0;
        for (size_t i = 1; i < n; i++) {
            double gap = (double)dt_days_between(orders[i].first_time,
                                                 orders[i - 1].first_time);
            gap_sum += gap;
            if (gap > gap_max) {
                gap_max = gap;
            }
        }
        gap_mean = gap_sum / (double)(n - 1);

        if (n > 2) {
            double ss = 0.0;
            for (size_t i = 1; i < n; i++) {
                double gap = (double)dt_days_between(orders[i].first_time,
                                                     orders[i - 1].first_time);
                double d = gap - gap_mean;
                ss += d * d;
            }
            gap_std = sqrt(ss / (double)(n - 2));
        }
    }

    double recency_ratio = 1.0;
    if (gap_mean != 0.0) {
        recency_ratio = recency / gap_mean;
    }
    if (recency_ratio > RECENCY_RATIO_MAX) {
        recency_ratio = RECENCY_RATIO_MAX;
    }

    x[F_RECENCY] = recency;
    x[F_FREQUENCY] = (double)n;
    x[F_MONETARY] = sum_value;
    x[F_AVG_ORDER_VALUE] = mean_value;
    x[F_STD_ORDER_VALUE] = std_value;
    x[F_MAX_ORDER_VALUE] = max_value;
    x[F_TOTAL_ITEMS] = items;
    x[F_TENURE] = tenure;
    x[F_ACTIVE_SPAN] = (double)dt_days_between(last_time, first_time);
    x[F_GAP_MEAN] = gap_mean;
    x[F_GAP_STD] = gap_std;
    x[F_GAP_MAX] = gap_max;
    x[F_RECENCY_RATIO] = recency_ratio;
    x[F_ORDERS_90D] = orders_90d;
    x[F_REVENUE_90D] = revenue_90d;
    x[F_SHARE_90D] = revenue_90d / sum_value;
    x[F_IS_UK] = (double)is_uk;
}


/* 상품 종류 수: (고객, 상품 코드) 순으로 정렬해서 코드가 바뀔 때마다 하나씩 셈 */
static void fill_n_products(const Transaction **rows, size_t n_obs, FeatureTable *table) {
    qsort(rows, n_obs, sizeof(*rows), compare_by_customer_stock);

    size_t c = 0;
    for (size_t i = 0; i < n_obs; i++) {
        while (table->items[c].customer_id != rows[i]->customer_id) {
            c++;
        }
        if (i == 0 || rows[i]->customer_id != rows[i - 1]->customer_id ||
            strcmp(rows[i]->stock_code, rows[i - 1]->stock_code) != 0) {
            table->items[c].x[F_N_PRODUCTS] += 1.0;
        }
    }
}

/* 반품 횟수 */
/* (고객, 송장) 순으로 정렬해서 서로 다른 반품 송장을 셈 */
/* 송장이 바뀔 때만 +1, 같은 송장의 여러 줄은 한 번으로 셈 */
static int fill_returns(const ReturnTable *returns, long long cutoff, FeatureTable *table) {
    const Return **rows = malloc(returns->count * sizeof(*rows));
    if (rows == NULL) {
        return 0;
    }

    size_t n = 0;
    for (size_t i = 0; i < returns->count; i++) {
        if (returns->items[i].timestamp < cutoff) {
            rows[n] = &returns->items[i];
            n++;
        }
    }
    qsort(rows, n, sizeof(*rows), compare_returns_by_customer_invoice);

    for (size_t i = 0; i < n; i++) {
        int is_new_invoice = (i == 0 || rows[i]->customer_id != rows[i - 1]->customer_id ||
                              strcmp(rows[i]->invoice, rows[i - 1]->invoice) != 0);
        if (!is_new_invoice) {
            continue;
        }
        /* 관측창에 구매가 없는 고객의 반품은 찾지 못하고 넘어감 */
        CustomerFeatures *f = bsearch(&rows[i]->customer_id, table->items, table->count,
                                      sizeof(CustomerFeatures), compare_feature_customer_id);
        if (f != NULL) {
            f->x[F_N_RETURNS] += 1.0;
        }
    }
    free(rows);

    for (size_t i = 0; i < table->count; i++) {
        double rate = table->items[i].x[F_N_RETURNS] / table->items[i].x[F_FREQUENCY];
        if (rate > 1.0) {
            rate = 1.0;
        }
        table->items[i].x[F_RETURN_RATE] = rate;
    }
    return 1;
}


/* 관측창 거래를 정렬해 주문으로 접음 */
/* 주문을 (고객, 날짜) 순으로 정렬함 */
/* 라벨의 고객 순서대로 구간(start ~ end)을 잘라 fill_order_features 를 호출함 */
/* 중간에 어디서 실패해도 새지 않게 하기 위해 실패 경로마다 이미 잡은 메모리를 free하도록 작성함 */
int features_build(const TransactionTable *tx, const ReturnTable *returns,
                   const LabelTable *labels, long long cutoff,
                   FeatureTable *table) {
    table->items = NULL;
    table->count = 0;

    size_t n_obs = 0;
    const Transaction **rows = transactions_sorted_observation(tx, cutoff, &n_obs);
    if (rows == NULL) {
        printf("메모리가 부족함: 관측창 거래 %zu건을 정렬하지 못함\n", n_obs);
        return 0;
    }

    size_t n_orders = 0;
    Order *orders = fold_orders(rows, n_obs, &n_orders);
    if (orders == NULL) {
        printf("메모리가 부족함: 주문 %zu건을 담지 못함\n", n_orders);
        free(rows);
        return 0;
    }
    qsort(orders, n_orders, sizeof(Order), compare_orders_by_customer_date);

    table->items = calloc(labels->count, sizeof(CustomerFeatures));
    if (table->items == NULL) {
        printf("메모리가 부족함: 고객 %zu명의 피처를 담지 못함\n", labels->count);
        free(orders);
        free(rows);
        return 0;
    }

    /* 라벨과 같은 고객 순서로 한 명씩 채움 */
    size_t start = 0;
    for (size_t c = 0; c < labels->count; c++) {
        size_t end = start;
        while (end < n_orders && orders[end].customer_id == labels->items[c].customer_id) {
            end++;
        }
        if (end == start) {
            printf("라벨의 고객 %d 의 주문이 없음\n", labels->items[c].customer_id);
            features_free(table);
            free(orders);
            free(rows);
            return 0;
        }

        CustomerFeatures *f = &table->items[c];
        f->customer_id = labels->items[c].customer_id;
        f->churn = labels->items[c].churn;
        f->is_new = labels->items[c].is_new;
        fill_order_features(&orders[start], end - start, cutoff, f->x);
        table->count++;
        start = end;
    }
    free(orders);

    fill_n_products(rows, n_obs, table);
    free(rows);

    if (!fill_returns(returns, cutoff, table)) {
        printf("메모리가 부족함: 반품 정렬\n");
        features_free(table);
        return 0;
    }
    return 1;
}

void features_free(FeatureTable *table) {
    free(table->items);
    table->items = NULL;
    table->count = 0;
}

/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 주문 단위로 먼저 접는 이유 */
/*    파이썬은 (고객, 송장) 으로 먼저 묶어 주문 금액, 수량, 주문 시각을 만든 뒤 고객별 피처를 구함 */
/*    거래 줄 단위로 평균이나 간격을 구하면 상품 줄 수가 많은 주문이 더 크게 반영되어 값이 달라짐 */
/*    C 도 같은 순서로 거래 줄 -> 주문(Order) -> 고객 피처 순으로 접음 */
/* */
/* 2. 경계 사례 (파이썬과 같은 규칙) */
/*    주문이 1번인 고객: 표준편차 0, 주문 간격이 없으므로 gap_mean 과 gap_max 는 tenure 로 채움 */
/*    표준편차는 n-1 로 나눔 (pandas 기본값), 주문 간격은 n-1 개라서 n-2 로 나눔 */
/*    gap_mean 이 0 이면 recency_ratio 는 1.0, 그 외에는 recency / gap_mean 을 50 에서 자름 */
/*    order_date 는 송장의 가장 이른 시각, is_uk 는 가장 늦은 줄의 국가 기준 */
/* */
/* 3. 관측창 밖은 읽지 않음 */
/*    거래도 반품도 timestamp < cutoff 인 줄만 모아서 사용함 (함수 맨 앞에서 걸러냄) */
/*    이 규칙을 어기면 오류도 경고도 없이 AUC 만 올라감 (누수) */
/* */
/* 4. 허용 오차를 둔 이유와 결과 */
/*    실수는 더하는 순서가 다르면 마지막 자릿수가 달라지므로 상대 오차 1e-9 이하면 같다고 봄 */
/*    4,944명 x 20개 모두 정답과 일치, 가장 큰 오차는 std_order_value 의 1.36e-13 */
/*    일수를 센 피처(recency, tenure, gap 등)와 개수 피처는 오차가 정확히 0 */
/* */
/* 5. 한계 (이번 단계에서 직접 겪음) */
/*    C 에는 고객 번호로 바로 찾는 딕셔너리(해시맵)가 없어서 정렬을 목적마다 다시 해야 했음 */
/*    거래를 (고객, 송장) 순, (고객, 상품) 순으로, 반품을 (고객, 송장) 순으로 정렬하고 */
/*    고객을 찾을 때는 이진 탐색을 씀 */
/*    이 데이터에서는 0.17초라 속도 문제는 아니고 같은 모양의 코드가 길어지는 것이 불편함 */
/* ==================================================================================================== */


