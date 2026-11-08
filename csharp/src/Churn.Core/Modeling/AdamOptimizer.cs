namespace Churn.Core.Modeling;

// Adam: 기울기를 그대로 쓰지 않고 이동 평균으로 다듬어서 가중치를 고침
// m: 기울기의 이동 평균 (방향을 안정시킴)
// v: 기울기 제곱의 이동평균 (기울기가 큰 가중치는 조금, 작은 가중치는 상대적으로 크게 움직이게 함)
// sklearn 의 Adam 과 동일한 식
public sealed class AdamOptimizer
{
    private readonly double _learningRate;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _epsilon;

    // 가중치마다 하나씩 있는 이동평균
    private readonly double[] _m;
    private readonly double[] _v;

    // 지금까지 고친 횟수
    private int _step;

    public AdamOptimizer(
        int count,
        double learningRate = 1e-3,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    )
    {
        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "가중치는 1개 이상이어야 함");
        }
        _m = new double[count];
        _v = new double[count];
        _learningRate = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
    }

    // parameters 를 gradients 방향의 반대로 한 걸음 고침
    // 제자리에서 바꾼 것
    public void Update(double[] parameters, double[] gradients)
    {
        if (parameters.Length != _m.Length || gradients.Length != _m.Length)
        {
            throw new ArgumentException("가중치와 기울기의 개수가 맞지 않음");
        }

        _step++;

        // 처음에는 m, v 가 0 에서 시작해 실제보다 작게 나옴
        // 걸음 수에 따라 줄어드는 보정을 한 번에 곱해 줌
        double stepSize = _learningRate * Math.Sqrt(1.0 - Math.Pow(_beta2, _step)) / (1.0 - Math.Pow(_beta1, _step));

        for (int k = 0; k < parameters.Length; k++)
        {
            _m[k] = _beta1 * _m[k] + (1.0 - _beta1) * gradients[k];
            _v[k] = _beta2 * _v[k] + (1.0 - _beta2) * gradients[k] * gradients[k];
            parameters[k] -= stepSize * _m[k] / (Math.Sqrt(_v[k]) + _epsilon);
        }
    }
}
