using Churn.Core.Evaluation;
using Churn.Core.Models;
using Churn.Core.Modeling;


namespace Churn.Tests;


// MetricsCalculator 의 시험
// 작은 데이터는 손으로 직접 계산한 값
// 실제 데이터는 tools/expected_values.py 의 출력과 비교하기 위해 작성함
public class MetricsCalculatorTests
{

    // 손으로 계산하기 쉬운 고객 6명을 선정함
    // 확률이 높은 순으로 선정하였음
    // 확률 0.9 이탈, 0.8 이탈, 0.7 유지, 0.6 이탈, 0.4 유지, 0.2 유지
    // 이탈 3명 * 유지 3명 = 9쌍
    // 이 9쌍 중 이탈의 확률이 더 높은 쌍은 8쌍이므로 AUC = 8/9
    private static readonly bool[] SampleChurn = { true, true, false, true, false, false };
    private static readonly double[] SampleProbability = { 0.9, 0.8, 0.7, 0.6, 0.4, 0.2 };

    // 임계값 0.5 이상을 이탈로 예측하면 0.9, 0.8, 0.7, 0.6 이 이탈 예측이 됨
    // TP 3(0.9, 0.8, 0.6)
    // FP 1(0.7)
    // Fn 0
    // TN 2
    // 정확도 5/6
    // 정밀도 3/4
    // 재현율 1
    // F1 = 2 * 0.75 * 1 / 1.75 = 6/7
    // Brier = 고객별 (확률 - 정답)^2 의 평균, 정답은 이탈 1 유지 0
    //   0.9 이탈 (0.9 - 1)^2 = 0.01, 0.8 이탈 0.04, 0.7 유지 (0.7 - 0)^2 = 0.49
    //   0.6 이탈 0.16, 0.4 유지 0.16, 0.2 유지 0.04
    //   (0.01 + 0.04 + 0.49 + 0.16 + 0.16 + 0.04) / 6 = 0.9 / 6 = 0.15
    // 생성자 ConfusionMatrix(2, 1, 0, 3) 의 순서는 Tn, Fp, Fn, Tp
    [Fact]
    public void Evaluate_computes_every_metric_for_a_small_sample()
  {
      var m = MetricsCalculator.Evaluate(SampleChurn, SampleProbability);

      Assert.Equal(new ConfusionMatrix(2, 1, 0, 3), m.Matrix);
      Assert.Equal(5.0 / 6.0, m.Accuracy, 12);
      Assert.Equal(0.75, m.Precision, 12);
      Assert.Equal(1.0, m.Recall, 12);
      Assert.Equal(6.0 / 7.0, m.F1, 12);
      Assert.Equal(0.15, m.Brier, 12);
      Assert.Equal(8.0 / 9.0, m.Auc, 12);
  }

  // PR-AUC: 확률이 높은 순으로 보며 재현율이 늘어난 만큼에 그때의 정밀도를 곱해 더함
  // 0.9 에서의 재현율 1/3 정밀도 1 -> 1/3
  // 0.8 에서의 재현율 2/3 정밀도 1 -> 1/3
  // 0.6 에서의 재현율 1 정밀도 3/4 -> 1/3 * 3/4 = 1/4
  // 합계 11/12
  [Fact]
  public void Evaluate_computes_pr_auc()
  {
      var m = MetricsCalculator.Evaluate(SampleChurn, SampleProbability);

      Assert.Equal(11.0 / 12.0, m.PrAuc, 12);
  } 

  // 임계값을 바꾸면 혼동행렬이 바뀜
  // 임계값을 바꾸어도 AUC, PR-AUC, Brier 는 그대로여야 함
  [Fact]
  public void Evaluate_changes_matrix_but_not_auc_with_threshold()
  {
      var low = MetricsCalculator.Evaluate(SampleChurn, SampleProbability, 0.3);
      var high = MetricsCalculator.Evaluate(SampleChurn, SampleProbability, 0.85);

      // 0.3 이상: 0.9 0.8 0.7 0.6 0.4 -> Tp 3, Fp 2
      Assert.Equal(new ConfusionMatrix(1, 2, 0, 3), low.Matrix);

      // 0.85 이상: 0.9 만 -> Tp 1, Fn 2
      Assert.Equal(new ConfusionMatrix(3, 0, 2, 1), high.Matrix);

      Assert.Equal(low.Auc, high.Auc);
      Assert.Equal(low.PrAuc, high.PrAuc);
      Assert.Equal(low.Brier, high.Brier); 
  }

  // 확률이 모두 같으면 순서를 정할 수 없으므로 AUC 는 정확히 0.5로 둠
  // PR-AUC 는 이탈률과 동일함
  // 이탈 4명 / 전체 10명
  [Fact]
  public void Evaluate_treats_equal_probabilities_as_ties()
  {
      var churn = new[] { true, true, true, true, false, false, false, false, false, false };
      var probability = new double[10];
      Array.Fill(probability, 0.5);

      var m = MetricsCalculator.Evaluate(churn, probability);

      Assert.Equal(0.5, m.Auc);
      Assert.Equal(0.4, m.PrAuc, 12);
  }

  // 이탈을 모두 위에 놓으면 1
  // 이탈을 모두 아래에 놓으면 0
  // perfect: 이탈 고객(앞의 2명)에게 높은 확률을 줌, reversed: 반대로 이탈 고객에게 낮은 확률을 줌
  [Fact]
  public void Evaluate_gives_one_for_perfect_and_zero_for_reversed_ranking()
  {
      var churn = new[] { true, true, false, false };

      var perfect = MetricsCalculator.Evaluate(churn, new[] { 0.9, 0.8, 0.2, 0.1 });
      var reversed = MetricsCalculator.Evaluate(churn, new[] { 0.1, 0.2, 0.8, 0.9 });

      Assert.Equal(1.0, perfect.Auc);
      Assert.Equal(0.0, reversed.Auc);
  }

  // 이탈로 예측한 고객이 한 명도 없으면 정밀도, F1 의 분모가 0이 됨
  // 오류 대신 0 으로 처리함
  // sklearn 의 zero_division=0 과 동일함
  [Fact]
  public void Evaluate_returns_zero_when_nobody_is_predicted_as_churn()
  {
      var m = MetricsCalculator.Evaluate(new[] { true, false }, new[] { 0.1, 0.2 });

      Assert.Equal(0.0, m.Precision);
      Assert.Equal(0.0, m.Recall);
      Assert.Equal(0.0, m.F1);
  }

  [Fact]
  public void Evaluate_rejects_empty_or_mismatched_input()
  {
      Assert.Throws<ArgumentException>(() => MetricsCalculator.Evaluate(new bool[0], new double[0]));
      Assert.Throws<ArgumentException>(() => MetricsCalculator.Evaluate(new[] { true }, new[] { 0.5, 0.6 }));
  }

  // 정답 확률(reference_test_pred.csv 의 p_logit)로 계산하면 sklearn 값과 동일해야 함
  // 기대값은 tools/expected_values.py 의 [reference_test_pred.csv, 임계값 0.5] 출력
  [DataFact]
  public void Evaluate_matches_sklearn_on_reference_predictions()
  {
      var (churn, probability) = TestData.ReadReferencePredictions();

      var m = MetricsCalculator.Evaluate(churn, probability);

      Assert.Equal(0.781833993130, m.Auc, 10);
      Assert.Equal(0.750203587894, m.PrAuc, 10);
      Assert.Equal(0.712938005391, m.Accuracy, 10);
      Assert.Equal(0.684278350515, m.Precision, 10);
      Assert.Equal(0.745786516854, m.Recall, 10);
      Assert.Equal(0.713709677419, m.F1, 10);
      Assert.Equal(0.190716936782, m.Brier, 10);
      Assert.Equal(new ConfusionMatrix(527, 245, 181, 531), m.Matrix);
  }

  // C# 이 학습한 모델의 확률로 계산하면 C 와 동일한 값
  // 정답 CSV 의 계수가 일찍 멈춘 값이라 혼동행렬이 1명씩 다르게 나옴
  // 위 시험(정답 확률) 527 245 181 531 -> 이 시험(C# 모델 확률) 526 246 180 532
  // 마지막 줄: 정답 AUC 0.781834 와의 차이가 1e-4 보다 작아야 함
  [DataFact]
  public void Evaluate_gives_same_result_as_c_for_trained_model()
  {
      var model = new LogisticRegression();
      model.Fit(TestData.Train);
      var churn = new List<bool>();
      var probability = new List<double>();
      foreach (var c in TestData.Test)
    {
        churn.Add(c.Churn);
        probability.Add(model.PredictProbability(c));
    }

    var m = MetricsCalculator.Evaluate(churn, probability);

    Assert.Equal(0.781821, m.Auc, 6);
    Assert.Equal(new ConfusionMatrix(526, 246, 180, 532), m.Matrix);
    Assert.InRange(Math.Abs(m.Auc - 0.781833993130), 0.0, 1e-4);
  }
}