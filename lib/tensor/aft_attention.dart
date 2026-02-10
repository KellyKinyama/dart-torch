import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class AFTAttention extends Module {
  final Linear key;
  final Linear query;
  final Linear value;
  final Tensor w; // Position bias [MaxSeqLen, MaxSeqLen]
  final int headSize;

  AFTAttention(int embedSize, this.headSize, int maxSeqLen)
      : key = Linear(embedSize, headSize),
        query = Linear(embedSize, headSize),
        value = Linear(embedSize, headSize),
        w = Tensor.random([maxSeqLen, maxSeqLen]);

  @override
  Tensor forward(Tensor x) {
    // x shape: [T, Dim]
    final T = x.shape[0];

    // 1. Projections [T, HeadSize]
    final k = key.forward(x);
    final q = query.forward(x);
    final v = value.forward(x);

    // 2. AFT-Full formula components
    final sigmoidQ = q.sigmoid();
    final expK = k.exp();
    final expKV = expK * v; // Element-wise product [T, HeadSize]

    // 3. Vectorized weighted sums
    // We use the position bias matrix 'w' to weight the entire sequence at once.
    // numerator = w * (exp(k) * v)  -> [T, T] matmul [T, HeadSize] = [T, HeadSize]
    // denominator = w * exp(k)     -> [T, T] matmul [T, HeadSize] = [T, HeadSize]

    // We slice 'w' to match the current sequence length T
    // (Assuming a simple slice of the first T rows/cols of the bias matrix)
    final contextNum = w.matmul(expKV);
    final contextDen = w.matmul(expK);

    // 4. Final output: Gating * (WeightedValues / WeightSums)
    return sigmoidQ * (contextNum / contextDen);
  }

  @override
  List<Tensor> parameters() => [
        ...key.parameters(),
        ...query.parameters(),
        ...value.parameters(),
        w,
      ];
}
