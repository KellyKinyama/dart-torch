import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// Attention Free Transformer (AFT-full) implementation.
class AFTAttention extends Module {
  final Layer key;
  final Layer query;
  final Layer value;
  final int headSize;

  // Learned pair-wise position biases
  // For a sequence of length T, this is often T x T, but here we
  // initialize for a max context or dynamic resizing.
  late List<ValueVector> w;

  AFTAttention(int embedSize, this.headSize, int maxSeqLen)
      : key = Layer.fromNeurons(embedSize, headSize),
        query = Layer.fromNeurons(embedSize, headSize),
        value = Layer.fromNeurons(embedSize, headSize) {
    // Initialize learned position biases
    w = List.generate(maxSeqLen,
        (_) => ValueVector(List.generate(maxSeqLen, (_) => Value(0.01))));
  }

  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;

    // Linear transformations
    final k = x.map((v) => key.forward(v)).toList();
    final q = x.map((v) => query.forward(v)).toList();
    final v = x.map((v) => value.forward(v)).toList();

    // Apply sigmoid to Query as per AFT formula
    final sigmoidQ = q.map((vec) => vec.sigmoid()).toList();

    final out = List.generate(T, (t) {
      // numerator = sum_{t'=1}^T exp(K_t' + w_{t,t'}) * V_t'
      // denominator = sum_{t'=1}^T exp(K_t' + w_{t,t'})

      ValueVector numerator = ValueVector(List.filled(headSize, Value(0.0)));
      ValueVector denominator = ValueVector(List.filled(headSize, Value(0.0)));

      for (int tp = 0; tp < T; tp++) {
        // Bias for this specific pair of positions
        final bias = w[t].values[tp];

        // Element-wise exp(K + w)
        final expKW =
            ValueVector(k[tp].values.map((kv) => (kv + bias).exp()).toList());

        numerator += (expKW * v[tp]);
        denominator += expKW;
      }

      // Element-wise division of reduced context
      final context = numerator / denominator;

      // Final output is element-wise product with Query
      return sigmoidQ[t] * context;
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...key.parameters(),
      ...query.parameters(),
      ...value.parameters(),
      ...w.expand((row) => row.values),
    ];
  }
}
