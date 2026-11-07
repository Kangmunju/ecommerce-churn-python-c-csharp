using Churn.Core.Evaluation;
using Churn.Core.Models;
using Churn.Core.Modeling;

namespace Churn.Core.Validation;

// 층화 k-fold 교차검증
// 고객을 k개 조각으로 나눠 한 조각씩 시험용으로 쓰고 나머지 조각으로 학습함
// 시험셋 한 번의 결과는 우연에 흔들릴 수 있어서 k번 반복하여 안정성을 확인함
// IClassifier 약속만 알면 되므로 로지스틱 회귀든 MLP 든 같은 코드로 돌릴 수 있음
public static class CrossValidator
{
    // customers: 전체 고객 (정답은 customers[i].Churn)
    // createModel: 학습하지 않은 새 모델을 만들어주는 함수
    //              조각마다 새 모델이 필요함 (앞 조각에서 학습한 결과가 남으면 안 됨)
    public static CrossValidationResult Run(
        IReadOnlyList<CustomerFeatures> customers,
        Func<IClassifier> createModel,
        int folds = 5,
        int seed = 42
    )
    {
        int n = customers.Count;

        var churn = new bool[n];
        for (int i = 0; i < n; i++)
        {
            churn[i] = customers[i].Churn;
        }
        var assignment = StratifiedFolds.Assign(churn, folds, seed);

        var oof = new double[n];
        var aucs = new double[folds];

        for (int fold = 0; fold < folds; fold++)
        {
            // 이번 조각에 들어있는 고객은 시험용
            // 나머지는 학습용
            var train = new List<CustomerFeatures>();
            var testIndexes = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if (assignment[i] == fold)
                {
                    testIndexes.Add(i);
                }
                else
                {
                    train.Add(customers[i]);
                }
            }

            // 표준화에 쓰는 평균과 표준편차도 학습용 고객으로만 구해짐
            // 모델의 Fit 안에서의 일
            var model = createModel();
            model.Fit(train);

            var foldChurn = new List<bool>(testIndexes.Count);
            var foldProbability = new List<double>(testIndexes.Count);
            foreach (int index in testIndexes)
            {
                double p = model.PredictProbability(customers[index]);
                oof[index] = p;
                foldChurn.Add(customers[index].Churn);
                foldProbability.Add(p);
            }

            // AUC 는 이 조각의 시험 고객들끼리만 비교해서 구함
            aucs[fold] = MetricsCalculator.Evaluate(foldChurn, foldProbability).Auc;
        }

        double sum = 0.0;
        foreach (double auc in aucs)
        {
            sum += auc;
        }
        double mean = sum / folds;

        double sumSquares = 0.0;
        foreach (double auc in aucs)
        {
            double d = auc - mean;
            sumSquares += d * d;
        }
        double std = Math.Sqrt(sumSquares / folds);

        return new CrossValidationResult(aucs, mean, std, oof);
    }
}

// ============================================================
// [설계 메모]
// ============================================================
//
// 1. 조각마다 새 모델을 만드는 이유 (누수 방지)
//    - 모델 하나를 계속 쓰면 앞 조각에서 학습한 계수가 남아서, 시험 고객을 이미 본 모델이 예측하게 될 수 있음
//    - 표준화의 평균과 표준편차도 Fit 안에서 구하기 때문에 조각마다 학습용 고객으로만 계산됨
//    - 그래서 모델을 직접 받지 않고 새 모델을 만드는 함수(Func<IClassifier>)를 받음
//
// 2. 조각이 파이썬과 다른 이유
//    - sklearn 의 섞는 방식을 C# 에서 그대로 재현할 수 없어서 StratifiedFolds 가 시드를 고정해 직접 나눔
//    - 같은 고객이 같은 조각에 들어가지 않으므로 조각별 AUC 와 표준편차는 파이썬과 다름
//    - 시드를 바꿔 가며 평균을 보면 파이썬의 평균이 그 범위 안에 들어와서, 시험은 평균만 범위로 비교함
//
// 3. OOF 확률을 따로 돌려주는 이유
//    - 고객마다 그 고객이 빠진 조각으로 학습한 모델의 예측이라서, 모든 고객의 확률이 처음 보는 데이터에 대한 것임
//    - 10단계에서 이 확률로 쿠폰 임계값을 고름 (시험셋에서 고르면 낙관적인 한계가 있음)
//
// 4. 표준편차를 n 으로 나누는 이유
//    - 파이썬 numpy 의 std 기본값이 n 으로 나누는 모표준편차라서 맞춤
//    - 조각이 5개뿐이라 n-1 로 나누면 값이 눈에 띄게 달라짐
