import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class AFTAttention extends Module {
  final Layer key;
  final Layer query;
  final Layer value;
  final int headSize;
  final bool masked;
  final Tensor w; // Flattened position bias [maxSeqLen, maxSeqLen]

  AFTAttention(int embedSize, this.headSize, int maxSeqLen,
      {this.masked = false})
      : key = Layer(embedSize, headSize, useGelu: false),
        query = Layer(embedSize, headSize, useGelu: false),
        value = Layer(embedSize, headSize, useGelu: false),
        // Initialize position biases with small random values
        w = Tensor.random([maxSeqLen, maxSeqLen]);

  Tensor forward(Tensor x) {
    final int T = x.shape[0];
    final int d = headSize;

    // 1. Projections
    final k = key.forward(x);
    final q = query.forward(x).sigmoid(); // sigma_q
    final v = value.forward(x);

    // Final output buffer
    final out = Tensor([T, d], children: {k, q, v, w});

    // 2. AFT-Full Logic
    for (int t = 0; t < T; t++) {
      Tensor numerator = Tensor.zeros([1, d]);
      Tensor denominator = Tensor.zeros([1, d]);

      // Causality check
      int endRange = masked ? t + 1 : T;

      for (int tp = 0; tp < endRange; tp++) {
        // Access w_t,tp from the flat bias data
        double bias = w.data[t * w.shape[1] + tp];

        // exp(K_tp + bias)
        // This is the element-wise exponential part of the paper
        final expKW = (k.getRow(tp) + bias).exp();

        numerator = numerator + (expKW * v.getRow(tp));
        denominator = denominator + expKW;
      }

      // Normalization and Gating
      final context = numerator / (denominator + 1e-9);
      out.setRow(t, q.getRow(t) * context);
    }

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...key.parameters(),
        ...query.parameters(),
        ...value.parameters(),
        w,
      ];
}
