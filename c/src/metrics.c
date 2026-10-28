#include "metrics.h"


#include <stdio.h>
#include <stdlib.h>


/* 확률과 정답을 한 쌍으로 묶은 것 */
/* 확률을 기준으로 정렬하면서 정답이 같이 따라다니게 하려고 작성함 */
typedef struct 
{
    double prob;
    int y;
} Scored;


/* 확률 배열만 정렬하면 어느 확률이 어느 정답의 것인지 잃게 되므로 (확률, 정답) 쌍을 Scored로 묶어 정렬함 */
/* 확률이 큰 쪽이 앞에 오도록 비교함 */
/* qsort 용 */
static int compare_by_prob_desc(const void *a, const void *b) {
    const Scored *x = a;
    const Scored *y = b;
    if (x->prob > y->prob) {
        return -1;
    }
    if (x->prob < y->prob) {
        return 1;
    }
    return 0;
}


/* 분모가 0이면 0으로 돌려줌 */
/* sklearn 의 zero_division=0과 동일함 */
static double safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}


/* 확률이 높은 순으로 정렬된 목록을 훑으며 AUC 와 PR-AUC 를 함께 구함 */
/* 같은 확률을 가진 고객들은 한 묶음으로 보고 묶음 단위로 계산함 */
/* total_pos : 전체 이탈 고객 수 */
/* total_neg : 유지 고객 수 */
/* AUC : 이탈 고객이 나올 때마다 뒤(확률이 더 낮은 쪽)에 유지 고객이 몇 명 있는지를 세어 더함 */
/* PR-AUC : 줄이 늘어난 구간마다 그때의 정밀도를 곱해 더함 */
static void compute_auc_and_pr_auc(const Scored *sorted, size_t n, long long total_pos, long long total_neg, double *auc, double *pr_auc) {

  /* seen_pos : 앞선 묶음들(더 높은 확률에 있던 이탈) */
  long long seen_pos = 0;

  /* seen_neg : 유지 고객 수 */
  long long seen_neg = 0;

  /* pair_score : 이탈 고객과 유지 고객을 하나씩 짝지었을 때 이탈 고객의 확률이 더 높은 쌍의 수 */
  /*              확률이 동일한 쌍은 0.5 쌍으로 셈 */
  double pair_score = 0.0;
  double ap = 0.0;
  double prev_recall = 0.0;

  size_t i = 0;
  while (i < n) {

    /* 확률이 sorted[i] 와 같은 고객들이 한 묶음 */
    size_t j = i;
    long long group_pos = 0;
    long long group_neg = 0;
    while(j < n && sorted[j].prob == sorted[i].prob) {
        if (sorted[j].y == 1) {
            group_pos++;
        } else {
            group_neg++;
        }
        j++;
    }

    /* 이 묶음의 이탈 고객이 이기는 유지 고객 수 */
    /* 확률이 더 낮은 유지 고객(전체 - 앞선 묶음 - 이 묶음)은 1쌍씩 */
    /* 확률이 같은 유지 고객(이 묶음)은 0.5 쌍씩 */
    long long lower_neg = total_neg - seen_neg - group_neg;
    pair_score += (double)group_pos * ((double)lower_neg + 0.5 * (double)group_neg);

    /* 이 묶음까지를 이탈로 예측했을 때의 정밀도와 재현율을 계산함 */
    seen_pos += group_pos;
    seen_neg += group_neg;
    double precision = safe_divide((double)seen_pos, (double)(seen_pos + seen_neg));
    double recall = safe_divide((double)seen_pos, (double)total_pos);
    
    /* 재현율이 늘어난 만큼에 그때의 정밀도를 곱해서 더함 */
    ap += (recall - prev_recall) * precision;
    prev_recall = recall;

    i = j;
  }

  *auc = safe_divide(pair_score, (double)total_pos * (double)total_neg);
  *pr_auc = ap;
  
}


int metrics_evaluate(const int *y, const double *prob, size_t n, double threshold, Metrics *out) {
    if(n == 0) {
        printf("평가할 고객이 없음\n");
        return 0;
    }
    Scored *sorted = malloc(n * sizeof(Scored));
    if (sorted == NULL) {
        printf("메모리 할당 실패\n");
        return 0;
    }

    /* 혼동행렬, Brier 점수, 정렬용 목록을 한 번에 만듦 */
    ConfusionMatrix cm = {0, 0, 0, 0};
    double brier_sum = 0.0;
    long long total_pos = 0;
    long long total_neg = 0;
    for (size_t i = 0 ; i < n ; i++) {
        sorted[i].prob = prob[i];
        sorted[i].y = y[i];

        int predicted = 0;
        if (prob[i] >= threshold) {
            predicted = 1;
        }
        if (y[i] == 1) {
            total_pos++;
            if (predicted == 1) {
                cm.tp++;
            } else {
                cm.fn++;
            }
        } else {
            total_neg++;
            if (predicted == 1) {
                cm.fp++;
            } else {
                cm.tn++;
            }
        }

        double diff = prob[i] - (double)y[i];
        brier_sum += diff * diff;
    }

    qsort(sorted, n, sizeof(Scored), compare_by_prob_desc);
    compute_auc_and_pr_auc(sorted, n, total_pos, total_neg, &out->auc, &out->pr_auc);
    free(sorted);

    out->cm = cm;
    out->accuracy = (double)(cm.tp + cm.tn) / (double)n;
    out->precision = safe_divide((double)cm.tp, (double)(cm.tp + cm.fp));
    out->recall = safe_divide((double)cm.tp, (double)(cm.tp + cm.fn));
    out->f1 = safe_divide(2.0 * out->precision * out->recall, out->precision + out->recall);
    out->brier = brier_sum / (double)n;
    return 1;
}


/* ==================================================================================================== */
/* [설계 메모] */
/* ==================================================================================================== */
/* 1. AUC 를 정렬 한 번으로 구하는 이유 */
/*    정의대로 하면 이탈 고객 x 유지 고객의 모든 쌍을 비교해야 해서 시험 고객 1,484명 기준 약 54만 번 비교함 */
/*    확률이 높은 순으로 한 번 정렬하고 앞에서부터 훑으면 이탈 고객마다 "뒤에 있는 유지 고객 수" 만 세면 되어 */
/*    정렬 비용(n log n) 만 듦, 고객이 수십만 명으로 늘어도 같은 방식으로 계산 가능함 */
/*    PR-AUC 도 같은 정렬 결과를 훑으면서 함께 구하므로 정렬은 한 번만 함 */
/* */
/* 2. 같은 확률은 한 묶음으로 처리함 */
/*    확률이 정확히 같은 고객들 사이에 순서를 매기면 정렬 방식에 따라 AUC 가 달라짐 */
/*    그래서 같은 확률의 고객은 묶음으로 보고, 묶음 안의 이탈 x 유지 쌍은 0.5 쌍씩 셈 (sklearn 과 같은 규칙) */
/*    PR-AUC 도 묶음이 끝날 때마다 정밀도와 재현율을 계산함 */
/*    확률을 전부 0.5 로 준 시험에서 AUC 가 정확히 0.5, PR-AUC 가 이탈률(0.4798) 로 나오는지 확인함 */
/* */
/* 3. 혼동행렬 칸의 순서와 임계값 */
/*    TN, FP, FN, TP 순서이며 파이썬 confusion_matrix 의 [[TN, FP], [FN, TP]] 와 같음 */
/*    정답 확률로 계산하면 TN 527, FP 245, FN 181, TP 531 이고 C 가 학습한 모델의 확률로 계산하면 526, 246, 180, 532 로 1명씩 다름 */
/*    정답 CSV 의 계수가 sklearn 이 일찍 멈춘 값이라 확률이 최대 8e-3 달랐고, 0.5 근처에 있던 고객이 반대편으로 넘어간 것임 */
/*    (logistic.c 하단 설계 메모 4 참고) */
/*    AUC, PR-AUC, Brier 는 확률 자체로 계산해 임계값의 영향을 받지 않고, 나머지 지표는 임계값 이상을 이탈로 봄 */
/* ==================================================================================================== */