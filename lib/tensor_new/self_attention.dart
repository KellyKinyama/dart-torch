import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class SelfAttention extends Module {
  final Layer queryLayer;
  final Layer keyLayer;
  final Layer valueLayer;
  final int headSize;

  SelfAttention(int embedSize, this.headSize)
      : queryLayer = Layer(embedSize, headSize, useGelu: false),
        keyLayer = Layer(embedSize, headSize, useGelu: false),
        valueLayer = Layer(embedSize, headSize, useGelu: false);

  Tensor forward(Tensor x, {bool masked = false}) {
    int T = x.shape[0]; // Sequence length

    // 1. Linear projections: Q, K, V
    Tensor q = queryLayer.forward(x);
    Tensor k = keyLayer.forward(x);
    Tensor v = valueLayer.forward(x);

    // 2. Scaled Dot-Product Attention: (Q @ K^T) / sqrt(dk)
    Tensor kt = k.transpose();
    Tensor scores = q.matmul(kt) * (1.0 / math.sqrt(headSize));

    // 3. Apply Causal Mask
    // We modify the data of the scores tensor directly before Softmax.
    // Since this happens after the matmul, the gradients for Q and K
    // will still flow correctly for the unmasked parts.
    if (masked) {
      for (int i = 0; i < T; i++) {
        for (int j = i + 1; j < T; j++) {
          // Setting future tokens to -infinity (or very large negative)
          // ensures their softmax probability is exactly 0.
          scores.data[i * T + j] = -1e15;
        }
      }
    }

    // 4. Attention mechanism
    //
    Tensor probs = scores.softmax();
    Tensor out = probs.matmul(v);

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...queryLayer.parameters(),
        ...keyLayer.parameters(),
        ...valueLayer.parameters()
      ];
}
