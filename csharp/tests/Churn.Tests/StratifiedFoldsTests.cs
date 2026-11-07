using Churn.Core.Validation;

namespace Churn.Tests;

// StratifiedFolds 의 시험
// 조각마다 크기와 이탈 고객 수가 최대 1명 차이로 고르게 나뉘는지 확인함
public class StratifiedFoldsTests
{
    // 이탈 positives 명 뒤에 유지 negatives 명을 붙인 정답 목록
    private static bool[] MakeChurn(int positives, int negatives)
    {
        var churn = new bool[positives + negatives];
        for (int i = 0; i < positives; i++)
        {
            churn[i] = true;
        }
        return churn;
    }

    // 조각 번호마다 고객 수와 이탈 고객 수를 셈
    // sizes[f]: f번 조각의 고객 수
    // positives[f]: f번 조각의 이탈 고객 수
    // 이름 붙은 튜플로 두 배열을 같이 돌려줌 (받을 때는 var (sizes, positives) = Count(...))
    private static (int[] Sizes, int[] Positives) Count(int[] assignment, bool[] churn, int folds)
    {
        var sizes = new int[folds];
        var positives = new int[folds];
        for (int i = 0; i < assignment.Length; i++)
        {
            sizes[assignment[i]]++;
            if (churn[i])
            {
                positives[assignment[i]]++;
            }
        }
        return (sizes, positives);
    }

    // 이탈 23명, 유지 17명을 5조각으로 나눔
    // 이탈은 5 5 5 4 4 명으로 나뉘는 것을 확인함
    // 유지는 이어서 배정되기 때문에 3 3 3 4 4 명으로 나뉘는 것을 확인함
    // 합치면 조각마다 정확히 8명이 됨
    [Fact]
    public void Assign_balances_sizes_and_churn_counts()
    {
        var churn = MakeChurn(23, 17);

        var assignment = StratifiedFolds.Assign(churn, 5, 42);
        var (sizes, positives) = Count(assignment, churn, 5);

        // 배열 안의 값이 순서까지 같은지 비교함
        Assert.Equal(new[] { 8, 8, 8, 8, 8 }, sizes);
        Assert.Equal(23, positives.Sum());
        Assert.True(positives.Max() - positives.Min() <= 1);
    }

    [Fact]
    public void Assign_puts_every_customer_in_a_valid_fold()
    {
        var assignment = StratifiedFolds.Assign(MakeChurn(23, 17), 5, 42);

        Assert.Equal(40, assignment.Length);
        Assert.All(assignment, fold => Assert.InRange(fold, 0, 4));
    }

    // 같은 seed 면 같은 결과가 나옴
    [Fact]
    public void Assign_is_repeatable_for_the_same_seed()
    {
        var churn = MakeChurn(23, 17);

        var first = StratifiedFolds.Assign(churn, 5, 42);
        var again = StratifiedFolds.Assign(churn, 5, 42);
        var other = StratifiedFolds.Assign(churn, 5, 7);

        Assert.Equal(first, again);
        Assert.NotEqual(first, other);
    }

    // Assert.Throws<T> 는 정확히 T 오류가 나야 통과함
    // ArgumentOutOfRangeException 은 ArgumentException 의 자식이라 서로 다른 시험으로 나눔
    [Fact]
    public void Assign_rejects_fewer_than_two_folds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => StratifiedFolds.Assign(MakeChurn(10, 10), 1, 42));
    }

    // 이탈 고객이 3명 뿐이면 5조각 중 두 조각에는 이탈 고객이 없어 AUC 를 구할 수 없음
    [Fact]
    public void Assign_rejects_when_a_class_is_smaller_than_folds()
    {
        Assert.Throws<ArgumentException>(() => StratifiedFolds.Assign(MakeChurn(3, 20), 5, 42));
        Assert.Throws<ArgumentException>(() => StratifiedFolds.Assign(MakeChurn(20, 3), 5, 42));
    }

    // 실제 고객 4,944명(이탈 2,372명)을 5조각으로 나눔
    // 조각 크기는 988 또는 989로 계산됨
    // 이탈 고객은 474 또는 475명으로 계산됨
    [DataFact]
    public void Assign_balances_real_customers()
    {
        var all = TestData.Train.Concat(TestData.Test).OrderBy(c => c.CustomerId).ToList();
        var churn = all.Select(c => c.Churn).ToArray();

        var assignment = StratifiedFolds.Assign(churn, 5, 42);
        var (sizes, positives) = Count(assignment, churn, 5);

        Assert.Equal(4944, sizes.Sum());
        Assert.Equal(2372, positives.Sum());
        Assert.All(sizes, size => Assert.InRange(size, 988, 989));
        Assert.All(positives, count => Assert.InRange(count, 474, 475));
    }
}
