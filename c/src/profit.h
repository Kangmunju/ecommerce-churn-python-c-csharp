#ifndef PROFIT_H
#define PROFIT_H


#include <stddef.h>


#include "labels.h"
#include "transactions.h"


/* 임계값 개수 */
/* 0.05 부터 0.95 까지 0.05 간격으로 설정*/
#define PROFIT_THRESHOLDS 19

/* 쿠폰 발송에 대한 가정 */
/* 실무에서는 재무팀, 마케팅팀에서 받는 숫자 */
typedef struct 
{
    
  /* 공헌 이익률 : 매출 중 이익으로 남는 비율 */
  double margin;

  /* 쿠폰 1장의 원가 */
  double coupon_cost;

  /* 이탈 예정 고객이 쿠폰을 받고 돌아올 확률 */
  double response;
} ProfitAssumptions;


/* 쿠폰 1장을 보냈을 때의 이익 */
/* 아무것도 보내지 않은 고객(FN, TN)은 이익도 손실도 0으로 체크함 */
typedef struct 
{
  
    /* 진짜 이탈할 고객에게 보냈을 때 (TP) */
    double value_tp;

    /* 어차피 남을 고객에게 보냈을 때 (FP) */
    double value_fp;
} ProfitValues;


/* 유지 고객 한명이 예측창에서 쓴 금액에 대한 실측값 */
typedef struct 
{
    double median;
    double mean;
    size_t count;
} RetainedRevenue;


/* 임계값 하나에 대한 결과 */
typedef struct 
{
    double threshold;

    /* 쿠폰을 보낸 고객 수 = tp + fp */
    long long sent;
    long long tp;
    long long fp;

    /* 보낸 고객 중 진짜 이탈 비율 */
    /* 보낸 고객이 없으면 0으로 처리함 */
    double precision;
    double profit;
} ProfitPoint;


/* 유지 고객(관측창과 예측창 모두 구매한 고객)의 예측창 매출을 재서 중앙값, 평균, 인원을 구함 */
/* 예측창: cutoff <= timestamp < label_end */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패 시 이유를 화면에 출력함 */
int profit_measure_retained(const TransactionTable *tx, const LabelTable *labels, long long cutoff, long long label_end, RetainedRevenue *out);

/* 가정과 유지 고객의 매출 중앙값으로 쿠폰 1장의 이익을 구함 */
ProfitValues profit_values(const ProfitAssumptions *assumptions, double r_retained);

/* 정답 y(이탈 1)와 확률 prob 로 임계값 19개의 이익 곡선을 채움 */
void profit_curve(const int *y, const double *prob, size_t n, const ProfitValues *values, ProfitPoint curve[PROFIT_THRESHOLDS]);

/* 곡선에서 이익이 가장 큰 임계값의 위치를 돌려줌 */
/* 같으면 앞쪽을 리턴함 */
int profit_best_index(const ProfitPoint curve[PROFIT_THRESHOLDS]);


#endif



