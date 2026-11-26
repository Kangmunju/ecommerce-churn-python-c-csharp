using System.Diagnostics;
using Churn.Core.Data;
using Churn.Core.Evaluation;
using Churn.Core.Features;
using Churn.Core.Labels;
using Churn.Core.Models;
using Churn.Core.Modeling;
using Churn.Core.Validation;


namespace Churn.Core.Pipeline;


// 1 ~ 10단계에서 만든 부품을 순서대로 이어서 처음부터 끝까지 한 번 돌림
// CSV 읽기 -> 이탈 라벨 -> 피처 20개 -> 학습/시험 나누기 -> 로지스틱 -> 평가 지표
// -> 기대이익 -> OOF 임계값 -> 확률 보정 -> 민감도 -> 교차검증 -> MLP
// 계산은 부품이 하고 여기서는 연결과 시간을 재는 것만 함
public static class PipelineRunner
{
    public static PipelineResult Run(PipelineOptions options)
    {
        var times = new List<StageTime>();
        var watch = Stopwatch.StartNew();

        // 바깥의 times 와 watch 를 그대로 쓸 수 있도록 지역 함수로 선언함
        // 호출하게 되면 지금까지 걸린 시간을 기록하고 시계를 0 으로 되돌리도록 작성함
        void Lap(string name)
        {
            times.Add(new StageTime(name, watch.Elapsed.TotalSeconds));
            watch.Restart();
        }

        // 1. 읽기
        var transactions = CsvLoader.LoadTransactions(Path.Combine(options.ExportDir, "clean_transactions.csv"));
        var returns = CsvLoader.LoadReturns(Path.Combine(options.ExportDir, "returns.csv"));
        Lap("CSV 읽기");

        // 2. 이탈 라벨
        var labels = LabelBuilder.Build(transactions, options.Cutoff, options.LabelEnd);
        Lap("이탈 라벨");

        // 3. 피처 20개
        var features = FeatureBuilder.Build(transactions, returns, labels, options.Cutoff);
        Lap("피처 20개");

        // 4. 학습, 시험 나누기
        // 파이썬의 무작위 분할 결과를 파일에서 읽음
        var (train, test) = Split(features, Path.Combine(options.ExportDir, "reference_split.csv"));
        var trainChurn = train.Select(c => c.Churn).ToList();
        var testChurn = test.Select(c => c.Churn).ToList();
        Lap("학습/시험 나누기");

        // 5. 로지스틱 학습 후 시험 고객의 확률과 평가 지표
        var logistic = new LogisticRegression();
        logistic.Fit(train);
        var testProbabilities = test.Select(c => logistic.PredictProbability(c)).ToList();
        var metrics = MetricsCalculator.Evaluate(testChurn, testProbabilities);
        Lap("로지스틱 학습과 지표");

        // 6. 기대이익
        // 시험 고객에서 고른 최적과 기본 0.50, 학습 고객의 OOF 로 고른 임계값
        var retained = ProfitCalculator.MeasureRetained(transactions, labels, options.Cutoff, options.LabelEnd);
        var values = ProfitCalculator.Values(options.Assumptions, retained.Median);
        var curve = ProfitCalculator.Curve(testChurn, testProbabilities, values);
        var best = curve[ProfitCalculator.BestIndex(curve)];
        var atHalf = ThresholdSelector.Apply(testChurn, testProbabilities, values, 0.50);
        var oof = CrossValidator.Run(train, () => new LogisticRegression(), 5, 42).OofProbabilities;
        double oofThreshold = ThresholdSelector.ChooseFromOof(trainChurn, oof, values);
        var oofApplied = ThresholdSelector.Apply(testChurn, testProbabilities, values, oofThreshold);
        Lap("기대이익과 OOF 임계값");

        // 7. 확률 보정 확인, 가정 민감도
        var calibration = CalibrationCalculator.Compute(testChurn, testProbabilities);
        var sensitivity = SensitivityAnalyzer.Analyze(
            testChurn, testProbabilities, retained.Median, options.Assumptions,
            new[] { 0.03, 0.05, 0.10, 0.20 }, new[] { 2.0, 5.0, 15.0 });
        Lap("보정과 민감도");

        // 8. 전체 고객 5-fold 교차검증 (로지스틱)
        var logisticCv = CrossValidator.Run(features, () => new LogisticRegression(), 5, 42);
        Lap("로지스틱 교차검증");

        // 9. MLP
        // 교차검증(시드 0)과 시드마다 시험 AUC
        var mlpCv = CrossValidator.Run(features, () => new MlpClassifier(0), 5, 42);
        var seedAucs = new List<double>();
        for (int seed = 0; seed < options.MlpSeedCount; seed++)
        {
            var mlp = new MlpClassifier(seed);
            mlp.Fit(train);
            var probabilities = test.Select(c => mlp.PredictProbability(c)).ToList();
            seedAucs.Add(MetricsCalculator.Evaluate(testChurn, probabilities).Auc);
        }
        Lap("MLP 교차검증과 시드 반복");

        return new PipelineResult(
            Customers: labels.Count,
            Churned: labels.Count(l => l.Churn),
            NewCustomers: labels.Count(l => l.IsNew),
            TrainCount: train.Count,
            TestCount: test.Count,
            LogisticIterations: logistic.Iterations,
            Logistic: metrics,
            RetainedMedian: retained.Median,
            Curve: curve,
            Best: best,
            AtHalf: atHalf,
            OofThreshold: oofThreshold,
            OofApplied: oofApplied,
            Calibration: calibration,
            Sensitivity: sensitivity,
            LogisticCv: logisticCv,
            MlpCv: mlpCv,
            MlpSeedAucs: seedAucs,
            Times: times
        );
    }

    // reference_split.csv (customer_id, split) 의 split 이 test 인 고객을 시험용으로 둠
    // 나머지를 학습으로 나눔
    private static (List<CustomerFeatures> Train, List<CustomerFeatures> Test) Split(
        IReadOnlyList<CustomerFeatures> features,
        string splitPath
    )
    {
        var testIds = new HashSet<int>();
        foreach (var line in File.ReadLines(splitPath).Skip(1))
        {
            var fields = line.Split(',');
            if (fields[1] == "test")
            {
                testIds.Add(int.Parse(fields[0]));
            }
        }

        var train = new List<CustomerFeatures>();
        var test = new List<CustomerFeatures>();
        foreach (var customer in features)
        {
            if (testIds.Contains(customer.CustomerId))
            {
                test.Add(customer);
            }
            else
            {
                train.Add(customer);
            }
        }
        return (train, test);
    }
}


// ============================================================
// [설계 메모]
// ============================================================
// 1. 계산과 출력을 나눈 이유
//    C 의 main.c 는 계산과 printf 가 섞여 있어서 숫자를 시험으로 확인하기 어려웠음
//    여기서는 Run 이 결과(PipelineResult)만 돌려주고 화면 출력은 App 이 하기 때문에
//    PipelineRunnerTests 가 같은 결과 객체의 숫자를 직접 확인할 수 있음
//
// 2. 학습 고객과 전체 고객을 나눠 쓴 이유
//    OOF 임계값은 시험 고객을 보지 않고 골라야 해서 학습 고객 3,460명의 OOF 확률로 고름
//    교차검증 평균은 파이썬 cv_scores.csv 처럼 전체 고객 4,944명으로 구함
//    학습, 시험 나누기는 sklearn 의 무작위 분할을 C# 에서 재현하지 못해 reference_split.csv 를 읽어 씀 (C 와 동일)
//
// 3. 한계
//    단계를 순서대로 실행하고 병렬로 돌리지 않음
//    MLP 교차검증과 시드 반복이 전체 시간의 대부분을 차지함
//    파이썬의 sklearn 과 조각, 난수가 달라서 교차검증과 MLP 는 조각별 값이 아니라 평균만 범위로 비교함
