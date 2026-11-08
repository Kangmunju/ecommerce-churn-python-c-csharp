using Churn.Core.Modeling;

namespace Churn.Tests;

// AdamOptimizer 의 시험
// 첫 걸음은 기울기 크기와 상관없이 학습률(0.001) 정도만 움직인다는 점을 손계산과 맞춰 봄
public class AdamOptimizerTests
{
    // 기울기 0.5 로 첫 걸음을 시작함
    // m = 0.1 * 0.5 = 0.05
    // v = 0.001 * 0.25 = 0.00025
    // 걸음 크기 = 0.001 * sqrt(1 - 0.999) / (1 - 0.9) = 0.000316228
    // 움직임 = 0.000316228 * 0.05 / sqrt(0.00025) = 0.001 (엡실론 1e-8 만큼만 작아짐)
    [Fact]
    public void Update_first_step_moves_about_the_learning_rate()
    {
        var adam = new AdamOptimizer(1);
        var parameters = new[] { 1.0 };

        adam.Update(parameters, new[] { 0.5 });

        Assert.Equal(1.0 - 0.001, parameters[0], 6);
    }

    // 기울기가 100배 커도 첫 걸음은 같음
    // 기울기가 음수이면 반대 방향(값이 커지는 쪽)으로 움직임
    [Fact]
    public void Update_first_step_ignores_gradients_size_and_follows_its_sign()
    {
        var adam = new AdamOptimizer(3);
        var parameters = new[] { 1.0, 1.0, 1.0 };

        adam.Update(parameters, new[] { 0.5, 50.0, -0.5 });

        Assert.Equal(0.999, parameters[0], 6);
        Assert.Equal(0.999, parameters[1], 6);
        Assert.Equal(1.001, parameters[2], 6);
    }

    // 같은 기울기가 계속되면 걸음마다 약 0.001 씩 움직임
    // 두 번째 걸음: m = 0.095, v = 0.00049975, 걸음 크기 = 0.001 * sqrt(0.001999) / 0.19
    [Fact]
    public void Update_keeps_stepping_about_the_learning_rate_for_a_constant_gradients()
    {
        var adam = new AdamOptimizer(1);
        var parameters = new[] { 1.0 };

        adam.Update(parameters, new[] { 0.5 });
        adam.Update(parameters, new[] { 0.5 });

        Assert.Equal(1.0 - 0.002, parameters[0], 6);
    }

    // 기울기가 0 이면 움직이지 않음
    [Fact]
    public void Update_does_not_move_when_the_gradient_is_zero()
    {
        var adam = new AdamOptimizer(1);
        var parameters = new[] { 1.0 };

        adam.Update(parameters, new[] { 0.0 });

        Assert.Equal(1.0, parameters[0]);
    }

    [Fact]
    public void Update_rejects_a_wrong_number_of_values()
    {
        var adam = new AdamOptimizer(2);

        Assert.Throws<ArgumentException>(() => adam.Update(new[] { 1.0 }, new[] { 0.5, 0.5 }));
        Assert.Throws<ArgumentException>(() => adam.Update(new[] { 1.0, 1.0 }, new[] { 0.5 }));
    }

    [Fact]
    public void Constructor_rejects_zero_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdamOptimizer(0));
    }
}
