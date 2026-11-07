namespace Churn.Core.Validation;

// 고객을 folds 개의 조각(fold)으로 나눔
// 각 조각의 이탈 비율이 전체와 비슷하도록 이탈 고객과 유지 고객을 따로 섞어서 나눔 (층화)
// 같은 seed 면 항상 같은 결과가 나옴
// 파이썬 StratifiedKFold(shuffle=True) 와 동일한 성질
// 섞는 방식이 달라 같은 고객이 같은 조각에 들어가지는 않음
public static class StratifiedFolds
{
    // 돌려주는 값: assignment[i] = i번째 고객이 들어간 조각 번호 (0 ~ folds-1)
    // churn[i]: i번째 고객의 정답
    public static int[] Assign(IReadOnlyList<bool> churn, int folds, int seed)
    {
        if (folds < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(folds), "조각은 2개 이상이어야 함");
        }

        var positives = new List<int>();
        var negatives = new List<int>();
        for (int i = 0; i < churn.Count; i++)
        {
            if (churn[i])
            {
                positives.Add(i);
            }
            else
            {
                negatives.Add(i);
            }
        }

        // 한 조각에 이탈 고객이나 유지 고객이 없으면 AUC 를 구할 수 없음
        if (positives.Count < folds || negatives.Count < folds)
        {
            throw new ArgumentException(
                $"이탈 {positives.Count}명, 유지 {negatives.Count}명으로는 {folds}개 조각을 나눌 수 없음", nameof(churn)
            );
        }

        // seed 로 만든 난수 생성기
        // 같은 seed 인 경우 같은 순서로 섞임
        var random = new Random(seed);
        Shuffle(positives, random);
        Shuffle(negatives, random);

        // 섞은 순서대로 0, 1, 2, ... folds-1, 0, 1, ... 번 조각에 한 명씩 넣음
        // 이탈 고객을 넣고 이어서 유지 고객을 넣으므로 조각마다 크기도 최대 1명 차이로 맞춰짐
        var assignment = new int[churn.Count];
        int next = 0;
        foreach (int i in positives)
        {
            assignment[i] = next % folds;
            next++;
        }
        foreach (int i in negatives)
        {
            assignment[i] = next % folds;
            next++;
        }
        return assignment;
    }

    // 피셔-예이츠 섞기: 카드 한 벌을 섞는 것과 같음
    // 맨 뒤 칸부터 앞으로 한 칸씩 내려오면서 "이 칸에 들어갈 카드" 를 아직 안 뽑힌 앞쪽 카드 중에서 무작위로 뽑아 자리를 바꿈
    // 한 번 자리가 정해진 뒤쪽 칸은 다시 건드리지 않음
    // 모든 순서가 같은 확률로 나옴
    private static void Shuffle(List<int> items, Random random)
    {
        // i > 0: 마지막으로 남은 0번 칸은 바꿀 상대가 자기 자신뿐이라 돌 필요가 없음
        for (int i = items.Count - 1; i > 0; i--)
        {
            // 0 ~ i 번 칸 중 하나를 고름 (Next(i + 1) 은 0 이상 i 이하, i 자신이 나오면 제자리)
            int j = random.Next(i + 1);

            // 튜플로 두 칸의 값을 맞바꿈
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}
