using Churn.Core.Evaluation;
using Churn.Core.Models;
using Churn.Core.Modeling;
using Churn.Core.Validation;

namespace Churn.Tests;

// CrossValidator 의 시험
// 가짜 모델로 조각을 나누어 쓰는 규칙이 지켜지는 지 확인함
// 실제 데이터는 파이썬 CV AUC 와 비교함
public class CrossValidatorTests
{
    // 가짜 모델들이 무슨 일을 했는지 적어두는 기록장을 만들어 둠
    // 이 시험에서만 쓰기 때문에 시험 클래스 안에 private 로 둠
    private sealed class Log
    {
        // 만들어진 모델 수
        public int ModelsCreated;

        // 조각마다 학습에 쓴 고객 번호
        // 오른쪽 타입을 왼쪽에서 알 수 있어서 new() 로 줄여 씀
        public List<HashSet<int>> TrainIds = new();

        // 조각마다 예측한 고객 번호
        public List<List<int>> PredictedIds = new();
    }

    // 학습한 고객과 예측한 고객을 기록장에 기록함
    // 확률은 항상 0.5 를 돌려주는 가짜 모델
    // 가짜 모델은 진짜 모델 대신 기록을 남기거나 정해진 답을 주는 시험용 대역임
    private sealed class SpyClassifier : IClassifier
    {
        private readonly Log _log;

        public SpyClassifier(Log log)
        {
            _log = log;
            _log.ModelsCreated++;
        }

        public void Fit(IReadOnlyList<CustomerFeatures> train)
        {
            var ids = new HashSet<int>();
            foreach (var c in train)
            {
                ids.Add(c.CustomerId);
            }
            _log.TrainIds.Add(ids);
            _log.PredictedIds.Add(new List<int>());
        }

        public double PredictProbability(CustomerFeatures customer)
        {
            // 가장 최근에 Fit 한 조각의 목록에 적음
            // [^1] 은 뒤에서 첫 번째 원소를 가리킴 (list[list.Count - 1] 과 같음)
            _log.PredictedIds[^1].Add(customer.CustomerId);
            return 0.5;
        }
    }

    // 정답을 훔쳐보고 이탈이면 0.9, 유지면 0.1 을 돌려주는 가짜 모델
    private sealed class OracleClassifier : IClassifier
    {
        // 학습할 것이 없는 모델이라 본문을 비워 둠
        // IClassifier 가 Fit 을 요구하기 때문에 메서드는 있어야 하고, 빈 {} 는 아무 일도 하지 않는다는 뜻임
        public void Fit(IReadOnlyList<CustomerFeatures> train)
        {
        }

        public double PredictProbability(CustomerFeatures customer)
        {
            if (customer.Churn)
            {
                return 0.9;
            }
            return 0.1;
        }
    }

    // 고객 번호로 확률을 정하는 가짜 모델
    // 조각마다 AUC 가 제각각 나오도록 함
    private sealed class ScrambleClassifier : IClassifier
    {
        // 학습할 것이 없어서 비워 둠
        public void Fit(IReadOnlyList<CustomerFeatures> train)
        {
        }

        public double PredictProbability(CustomerFeatures customer)
        {
            return (customer.CustomerId * 37 % 100) / 100.0;
        }
    }

    // 고객 번호와 정답만 의미 있는 시험용 고객
    private static CustomerFeatures Make(int id, bool churn)
    {
        return new CustomerFeatures(id, new double[FeatureNames.All.Count], churn, false);
    }

    // 고객 번호 1 ~ 40
    // 앞의 23명이 이탈
    private static List<CustomerFeatures> Sample()
    {
        var customers = new List<CustomerFeatures>();
        for (int i = 1; i <= 40; i++)
        {
            customers.Add(Make(i, i <= 23));
        }
        return customers;
    }

    // 조각마다 새 모델을 만들어 줌
    // 학습 고객과 시험 고객이 겹치지 않도록 함
    // 모든 고객이 정확히 한 번 예측되도록 함
    [Fact]
    public void Run_keeps_train_and_test_apart_and_predicts_everyone_once()
    {
        var log = new Log();

        // () => new SpyClassifier(log) 는 새 모델을 만드는 함수
        // 같은 log 를 모든 가짜 모델에 넘기기 때문에 5개 모델이 한 기록장에 같이 적음
        CrossValidator.Run(Sample(), () => new SpyClassifier(log), 5, 42);

        Assert.Equal(5, log.ModelsCreated);
        Assert.Equal(5, log.TrainIds.Count);

        var predictedOnce = new HashSet<int>();
        for (int fold = 0; fold < 5; fold++)
        {
            var train = log.TrainIds[fold];
            var test = log.PredictedIds[fold];

            // 40명 중 8명이 시험
            // 40명 중 32명이 학습
            Assert.Equal(32, train.Count);
            Assert.Equal(8, test.Count);

            foreach (int id in test)
            {
                // 컬렉션에 값이 없어야 통과함
                Assert.DoesNotContain(id, train);

                // HashSet.Add 는 이미 있던 값이면 false 를 반환함
                // 두 번 예측된 고객을 잡기 위함
                // Assert.True 의 두 번째 값은 실패했을 때 같이 출력되는 메시지
                Assert.True(predictedOnce.Add(id), $"고객 {id} 가 두 번 예측됨");
            }
        }

        Assert.Equal(40, predictedOnce.Count);
    }

    // 완벽한 모델은 조각마다 AUC 1, OOF 확률은 입력 고객 순서와 동일한 순서로 나오는 것을 확인함
    [Fact]
    public void Run_gives_one_for_a_perfect_model_and_keeps_input_order()
    {
        var customers = Sample();

        var result = CrossValidator.Run(customers, () => new OracleClassifier(), 5, 42);

        Assert.Equal(5, result.FoldAucs.Count);
        Assert.All(result.FoldAucs, auc => Assert.Equal(1.0, auc));
        Assert.Equal(1.0, result.MeanAuc);
        Assert.Equal(0.0, result.StdAuc);

        // i번째 확률은 i번째 고객의 것이어야 함
        // 이탈이면 0.9, 유지면 0.1
        Assert.Equal(40, result.OofProbabilities.Count);
        for (int i = 0; i < customers.Count; i++)
        {
            double expected = 0.1;
            if (customers[i].Churn)
            {
                expected = 0.9;
            }
            Assert.Equal(expected, result.OofProbabilities[i]);
        }
    }

    // 확률이 모두 동일하면 순서를 정할 수 없으므로 AUC 는 정확히 0.5 가 나옴
    [Fact]
    public void Run_gives_half_for_a_model_that_cannot_rank()
    {
        var result = CrossValidator.Run(Sample(), () => new SpyClassifier(new Log()), 5, 42);

        Assert.All(result.FoldAucs, auc => Assert.Equal(0.5, auc));
    }

    // 평균과 표준편차가 조각별 AUC 로 계산한 값과 같게 나오는 지 시험 안에서 다시 계산해서 확인함
    // 표준편차는 n 으로 나눔
    // 파이썬 numpy 의 std 와 동일함
    [Fact]
    public void Run_summarizes_fold_aucs_with_mean_and_population_std()
    {
        var result = CrossValidator.Run(Sample(), () => new ScrambleClassifier(), 5, 42);

        double mean = result.FoldAucs.Average();
        double variance = result.FoldAucs.Sum(a => (a - mean) * (a - mean)) / result.FoldAucs.Count;

        // 마지막 12 는 소수점 12자리까지 같으면 통과
        Assert.Equal(mean, result.MeanAuc, 12);
        Assert.Equal(Math.Sqrt(variance), result.StdAuc, 12);
        Assert.True(result.StdAuc > 0.0);
    }

    // 같은 seed 인 경우 같은 결과가 나옴
    [Fact]
    public void Run_is_repeatable_for_the_same_seed()
    {
        var first = CrossValidator.Run(Sample(), () => new ScrambleClassifier(), 5, 42);
        var again = CrossValidator.Run(Sample(), () => new ScrambleClassifier(), 5, 42);

        Assert.Equal(first.FoldAucs, again.FoldAucs);
        Assert.Equal(first.OofProbabilities, again.OofProbabilities);
    }

    // 이탈 고객 3명 같이 조각을 나눌 수 없는 입력의 경우가 있음을 확인함
    // 이 경우 StratifiedFolds 의 오류가 그대로 전달됨
    [Fact]
    public void Run_rejects_when_folds_cannot_be_built()
    {
        var customers = new List<CustomerFeatures>();
        for (int i = 1; i <= 20; i++)
        {
            customers.Add(Make(i, i <= 3));
        }

        Assert.Throws<ArgumentException>(() => CrossValidator.Run(customers, () => new OracleClassifier(), 5, 42));
    }

    // 실제 고객 4,944명으로 로지스틱 회귀의 5-fold CV AUC 를 구함
    // 파이썬(sklearn StratifiedKFold, 시드 42) 은 평균 0.8044, 표준편차 0.0080
    // 조각을 나누는 방식이 달라 조각별 값이 다르지만 평균은 같은 범위 안에 있어야 함
    // 시드를 20개로 바꿔 봐도 평균이 0.8027 ~ 0.8061 이었으므로 파이썬 값 +-0.003 을 허용함
    [DataFact]
    public void Run_gives_cv_auc_close_to_python_on_real_data()
    {
        var all = TestData.Train.Concat(TestData.Test).OrderBy(c => c.CustomerId).ToList();

        var result = CrossValidator.Run(all, () => new LogisticRegression(), 5, 42);

        Assert.Equal(5, result.FoldAucs.Count);

        // InRange(값, 최소, 최대): 범위 안이면 통과
        Assert.InRange(result.MeanAuc, 0.8044 - 0.003, 0.8044 + 0.003);
        Assert.InRange(result.StdAuc, 0.001, 0.015);

        // OOF 확률은 고객 4,944명 모두 0 ~ 1 사이
        // 이 확률 전체로 구한 AUC 도 비슷한 범위에 있음을 확인함
        Assert.Equal(4944, result.OofProbabilities.Count);
        Assert.All(result.OofProbabilities, p => Assert.InRange(p, 0.0, 1.0));

        var churn = all.Select(c => c.Churn).ToList();
        double pooledAuc = MetricsCalculator.Evaluate(churn, result.OofProbabilities).Auc;
        Assert.InRange(pooledAuc, 0.8044 - 0.003, 0.8044 + 0.003);
    }
}
