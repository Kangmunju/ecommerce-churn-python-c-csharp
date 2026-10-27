#ifndef FEATURES_H
#define FEATURES_H

#include <stddef.h>

#include "labels.h"
#include "returns.h"
#include "transactions.h"


/* 피처 번호 */
/* reference_features.csv 의 열 순서와 같음 */
enum {
    F_RECENCY,
    F_FREQUENCY,
    F_MONETARY,
    F_AVG_ORDER_VALUE,
    F_STD_ORDER_VALUE,
    F_MAX_ORDER_VALUE,
    F_TOTAL_ITEMS,
    F_TENURE,
    F_ACTIVE_SPAN,
    F_GAP_MEAN,
    F_GAP_STD,
    F_GAP_MAX,
    F_RECENCY_RATIO,
    F_ORDERS_90D,
    F_REVENUE_90D,
    F_SHARE_90D,
    F_N_PRODUCTS,
    F_N_RETURNS,
    F_RETURN_RATE,
    F_IS_UK,
    FEATURE_COUNT
};


/* 피처 이름 */
/* 번호 순서와 같음 */
extern const char *const FEATURE_NAMES[FEATURE_COUNT];

/* 고객 한 명의 피처 20개와 정답 */
typedef struct {
    double x[FEATURE_COUNT];
    int customer_id;
    int churn;
    int is_new;
} CustomerFeatures;

/* 피처 전체를 담는 표 */
/* customer_id 오름차순, labels 와 같은 순서 */
typedef struct {
    CustomerFeatures *items;
    size_t count;
} FeatureTable;

/* 관측창(timestamp < cutoff)의 거래와 반품만 사용함 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패 시 이유를 화면에 출력함 */
int features_build(const TransactionTable *tx, const ReturnTable *returns,
                   const LabelTable *labels, long long cutoff,
                   FeatureTable *table);

void features_free(FeatureTable *table);

#endif
