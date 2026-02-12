import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class AFTAttention extends Module {
  final Layer queryLayer, keyLayer, valueLayer;
  final int headSize;
  final Tensor posBias; // Learned w matrix [maxSeqLen, maxSeqLen]
  final bool masked; // Added flag

  AFTAttention(int embedSize, this.headSize, int maxSeqLen,
      {this.masked = false})
      : queryLayer = Layer(embedSize, headSize, useGelu: false),
        keyLayer = Layer(embedSize, headSize, useGelu: false),
        valueLayer = Layer(embedSize, headSize, useGelu: false),
        posBias = Tensor.random([maxSeqLen, maxSeqLen]);

  Tensor forward(Tensor x) {
    final int T = x.shape[0];
    final int d = headSize;

    // 1. Projections
    final Q = queryLayer.forward(x).sigmoid();
    final K = keyLayer.forward(x);
    final V = valueLayer.forward(x);

    final out = Tensor([T, d], children: {Q, K, V, posBias});

    // 2. AFT-full Loop
    for (int t = 0; t < T; t++) {
      Tensor numerator = Tensor.zeros([1, d]);
      Tensor denominator = Tensor.zeros([1, d]);

      // If masked, we only look at past tokens (0 to t)
      // If not, we look at the whole sequence (0 to T)
      int limit = masked ? t + 1 : T;

      for (int tp = 0; tp < limit; tp++) {
        // Bias for this pair of positions
        double w_ttp = posBias.data[t * posBias.shape[1] + tp];

        // exp(K_tp + w_ttp)
        final expWeight = (K.getRow(tp) + w_ttp).exp();

        numerator = numerator + (expWeight * V.getRow(tp));
        denominator = denominator + expWeight;
      }

      // Y_t = Q_t * (num / den)
      final row_t = Q.getRow(t) * (numerator / (denominator + 1e-9));
      out.setRow(t, row_t);
    }

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...queryLayer.parameters(),
        ...keyLayer.parameters(),
        ...valueLayer.parameters(),
        posBias
      ];
}
