#ifndef LOGISTIC_H
#define LOGISTIC_H


#include <stddef.h>


#include "features.h"


/* 학습된 모델 */
/* 표준화에 쓴 평균과 표준편차, 표준화된 피처에 대한 계수, 절편을 가짐 */
typedef struct {
    double mean[FEATURE_COUNT];
    double scale[FEATURE_COUNT];
    double coef[FEATURE_COUNT];
    double intercept;
    int iterations;
} LogisticModel;


/* 고객 n명의 피처(원래 값)와 정답으로 모델을 학습함 */
/* rows 는 학습에 사용할 고객의 FeatureTable 안 위치 */
/* c_value 는 규제 세기의 역수를 나타냄(작을수록 규제가 강한 성질을 이용함) */
/* 돌려주는 값: 1 = 성공, 0 = 실패 */
/* 실패 시 이유를 화면에 출력함 */
int logistic_fit(const FeatureTable *table, const size_t *rows, size_t n, double c_value, LogisticModel *model);


/* 고객 한 명이 이탈할 확률 */
/* 원래 값을 넣으면 안에서 표준화하도록 작성함 */
double logistic_predict(const LogisticModel *model, const CustomerFeatures *customer);


#endif
