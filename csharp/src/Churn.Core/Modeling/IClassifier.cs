using Churn.Core.Models;


namespace Churn.Core.Modeling;


// 이탈 확률을 예측하는 모델이 지켜야 할 약속
// 로지스틱 회귀와 이후에 만들 MLP 가 같은 약속을 따르면 교차검증
// 평가 코드를 함께 쓸 수 있음
public interface IClassifier
{

    // 학습 고객의 피처와 정답(Churn)으로 모델을 학습시킴
    void Fit(IReadOnlyList<CustomerFeatures> train);

    // 고객 한 명이 이탈할 확률
    // 0 ~ 1 사이
    double PredictProbability(CustomerFeatures customer);
}