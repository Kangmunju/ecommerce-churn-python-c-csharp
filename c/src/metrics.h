#ifndef METRICS_H
#define METRICS_H


#include <stddef.h>


/* 혼동 행렬 : 실제와 예측을 짝지어 센 네 가지 개수 */
/* tn = 실제 유지, 예측 유지 */
/* fp = 실제 유지, 예측 이탈 */
/* fn = 실제 이탈, 예측 유지 */
/* tp = 실제 이탈, 예측 이탈 */
typedef struct 
{
    long long tn;
    long long fp;
    long long fn;
    long long tp;
} ConfusionMatrix;


/* 평가 지표 모음 */
typedef struct 
{
    double auc;
    double pr_auc;
    double accuracy;
    double precision;
    double recall;
    double f1;
    double brier;
    ConfusionMatrix cm;
} Metrics;


/* 고객 n명의 정답 y(이탈 1, 유지 0)와 예측 확률 prob 로 지표를 계산함 */
/* 확률이 threshold 이상이면 이탈로 예측한 것으로 보고 혼동행렬과 정확도, 정밀도, 재현율, F1 을 구함 */
/* AUC, PR-AUC, Brier 는 임계값과 관계없이 확률 자체로 구함 */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패 시 이유를 화면에 출력함 */
int metrics_evaluate(const int *y,  const double *prob, size_t n, double threshold, Metrics *out);


#endif



