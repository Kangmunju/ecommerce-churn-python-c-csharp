#include "profit.h"


#include <stdio.h>
#include <stdlib.h>


/* 고객 번호로 labels 에서 고객을 찾기 위한 비교 */
/* bsearch 용으로 작성함 */
/* key: 찾는 고객 번호 */
/* element: labels 의 한 칸 */
static int compare_customer_id(const void *key, const void *element) {
    int wanted = *(const int *)key;
    const CustomerLabel *label = element;
    if (wanted < label->customer_id) {
        return -1;
    }
    if (wanted > label->customer_id) {
        return 1;
    }
    return 0;
}


/* 실수 두 개를 오름차순으로 비교함 */
/* qsort 용 */
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) {
        return -1;
    }
    if (x > y) {
        return 1;
    }
    return 0;
}


int profit_measure_retained(const TransactionTable *tx, const LabelTable *labels, long long cutoff, long long label_end, RetainedRevenue *out) {

  /* revenue[i]: labels 의 i번째 고객이 예측창에서 쓴 금액의 합 */
  /* 합계를 쌓는 배열을 생성해야 하므로 메모리를 0으로 채워주는 calloc 을 사용함 */
  double *revenue = calloc(labels->count, sizeof(double));
  if (revenue == NULL) {
      printf("메모리 할당 실패\n");
      return 0;
  }

  /* 예측창의 거래를 훑으며 고객의 금액(수량 * 가격)에 더함 */
  /* 관측창에 구매가 없는 고객은 labels 에 없으므로 건너뛰도록 함 */
  for (size_t i = 0 ; i < tx->count ; i++) {
      const Transaction *t = &tx->items[i];
      if (t->timestamp < cutoff || t->timestamp >= label_end) {
          continue;
      }
      const CustomerLabel *found = bsearch(&t->customer_id, labels->items, labels->count, sizeof(CustomerLabel), compare_customer_id);
      if (found == NULL) {
          continue;
      }
      revenue[found - labels->items] += (double)t->quantity * t->price;
  }

  /* 예측창에 구매한 고객(churn == 0)의 금액만 모음 */
  double *kept = malloc(labels->count * sizeof(double));
  if (kept == NULL) {
      printf("메모리 할당 실패\n");
      free(revenue);
      return 0;
  }
  size_t n = 0;
  double sum = 0.0;
  for (size_t i = 0 ; i < labels->count ; i++) {
      if (labels->items[i].churn == 0) {
          kept[n] = revenue[i];
          sum += revenue[i];
          n++;
      }
  }
  free(revenue);
  if (n == 0) {
      printf("예측창에 구매한 고객이 없음\n");
      free(kept);
      return 0;
  }

  /* 정렬 후의 가운데 값을 중앙값으로 사용함 */
  /* 짝수 명인 경우 가운데 두 값의 평균을 중앙값으로 사용함 */
  qsort(kept, n, sizeof(double), compare_double);
  if (n % 2 == 1) {
      out->median = kept[n / 2];
  } else {
      out->median = (kept[n / 2 - 1] + kept[n / 2]) / 2.0;
  }
  out->mean = sum / (double)n;
  out->count = n;
  free(kept);
  return 1;
}


ProfitValues profit_values(const ProfitAssumptions *assumptions, double r_retained) {
    ProfitValues values;
    values.value_tp = assumptions->response * r_retained * assumptions->margin - assumptions->coupon_cost;
    values.value_fp = -assumptions->coupon_cost;
    return values;
}


/* 임계값을 0.05 부터 0.95 까지 바꿔 가며 "그 임계값으로 쿠폰을 보냈다면 얼마가 남는가" 를 계산함 */
/* y : 시험 고객의 정답 (이탈 1, 유지 0), prob : 시험 고객의 이탈 확률, n : 시험 고객 수 */
/* values : 쿠폰 1장의 이익 (맞게 보냈을 때 value_tp, 헛되이 보냈을 때 value_fp) */
/* curve : 결과를 채울 배열 (임계값 하나당 한 칸, 모두 19칸) */
/* 임계값을 낮추면 더 많은 고객에게 보내서 진짜 이탈자를 더 잡지만(tp 증가) 헛되이 보내는 것도 늘어남(fp 증가) */
/* 두 힘이 균형을 이루는 임계값이 가장 큰 이익을 내므로 곡선을 모두 계산한 뒤 profit_best_index 로 고름 */
void profit_curve(const int *y, const double *prob, size_t n, const ProfitValues *values, ProfitPoint curve[PROFIT_THRESHOLDS]) {
    for (int k = 0 ; k < PROFIT_THRESHOLDS ; k++) {

      /* 이번 칸의 임계값 : k = 0 이면 0.05, k = 1 이면 0.10, ... k = 18 이면 0.95 */
      /* 0.05 를 계속 더하면 0.15000000000000002 처럼 오차가 쌓일 수 있으므로 */
      /* 5씩 늘리는 정수(5, 10, ... 95)를 100으로 나눠서 매번 정확하게 계산함 */
      double threshold = (double)(5 * (k + 1)) / 100.0;

      /* tp : 쿠폰을 보냈고 실제로 이탈할 고객 수 (맞게 보냄) */
      /* fp : 쿠폰을 보냈지만 실제로는 남을 고객 수 (헛되이 보냄) */
      /* 확률이 임계값 이상이면 쿠폰을 보내는 고객이고 그 중 정답이 이탈(1)이면 tp, 유지(0)이면 fp */
      /* 확률이 임계값보다 낮은 고객은 보내지 않으므로 이익도 손실도 없어서 세지 않음 */
      long long tp = 0;
      long long fp = 0;
      for (size_t i = 0 ; i < n ; i++) {
          if (prob[i] >= threshold) {
              if (y[i] == 1) {
                  tp++;
              } else {
                  fp++;
              }
          }
      }

      /* sent : 쿠폰을 보낸 고객 수 = tp + fp */
      curve[k].threshold = threshold;
      curve[k].sent = tp + fp;
      curve[k].tp = tp;
      curve[k].fp = fp;

      /* precision : 보낸 고객 중 진짜 이탈할 고객의 비율 = tp / (tp + fp) */
      /* 한 명도 안 보냈으면 tp + fp 가 0 이라 나눌 수 없으므로 0 으로 둠 */
      curve[k].precision = 0.0;
      if (tp + fp > 0) {
          curve[k].precision = (double)tp / (double)(tp + fp);
      }

      /* profit : 기대이익 = 맞게 보낸 수 * 맞게 보낸 1건의 이익 + 헛되이 보낸 수 * 헛되이 보낸 1건의 손실 */
      /* value_fp 는 -5 처럼 음수로 들어 있으므로 더하면 손실이 빠지는 것과 같음 */
      curve[k].profit = (double)tp * values->value_tp + (double)fp * values->value_fp;
    }
}


int profit_best_index(const ProfitPoint curve[PROFIT_THRESHOLDS]) {
    int best = 0;
    for (int k = 1 ; k < PROFIT_THRESHOLDS ; k++) {
        if (curve[k].profit > curve[best].profit) {
            best = k;
        }
    }
    return best;
}


/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. 쿠폰 1장의 이익이 14.05 가 되는 이유 */
/*    진짜 이탈할 고객에게 보내면 10% 확률로 돌아오고, 돌아오면 평균적으로 635.01 파운드를 쓰고, 그 중 30% 가 이익임 */
/*    0.10 * 635.01 * 0.30 - 5 = 14.0503 */
/*    어차피 남을 고객에게 보내면 쿠폰 값 5 파운드만 잃음, 보내지 않은 고객은 이익도 손실도 0 */
/*    그래서 이익은 "맞게 보낸 수 * 14.05 - 헛되이 보낸 수 * 5" 한 줄로 정해짐 */
/* */
/* 2. 평균 대신 중앙값을 쓰는 이유 */
/*    유지 고객 2,572명의 예측창 매출은 중앙값 635.01, 평균 1758.29 로 크게 다름 */
/*    큰손 몇 명이 평균을 끌어올리므로 평균을 쓰면 쿠폰의 이익을 과대평가하게 됨 */
/*    한 명이 쓰는 금액으로는 중앙값이 더 보수적이고 현실에 가까움 */
/*    매출은 가정이 아니라 데이터에서 잰 값이고, 마진, 쿠폰 원가, 반응률은 가정임 (코드와 main.c 맨 위에 모아 둠) */
/* */
/* 3. 한계 (파이썬 a5_profit.py 에서 확인한 내용, C 에서는 아직 구현하지 않음) */
/*    임계값을 시험 고객에서 골랐기 때문에 이 이익은 낙관적일 수 있음 */
/*    파이썬은 학습 고객의 out-of-fold 확률(교차검증)에서 임계값을 고른 뒤 시험 고객에 적용해 확인했음 (0.25 에서 6,909 파운드) */
/*    교차검증, 확률 보정 확인, 가정이 틀렸을 때의 민감도 분석은 C# 단계에서 다룸 */
/*    쿠폰을 받아서 돌아온 것인지 원래 돌아올 사람인지는 관찰 데이터로 알 수 없어서 반응률은 가정으로 둘 수밖에 없음 */
/* ==================================================================================================== */