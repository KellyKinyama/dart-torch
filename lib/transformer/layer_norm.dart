import '../nn/value.dart';
import '../nn/value_vector.dart';

import 'dart:math';
class LayerNorm {
  final int size;
  final double eps;
  final List<Value> gamma;
  final List<Value> beta;

  LayerNorm(this.size, {this.eps = 1e-5})
      : gamma = List.generate(size, (_) => Value(1.0)),
        beta = List.generate(size, (_) => Value(0.0));

  ValueVector forward(ValueVector x) {
    final mean = x.values.reduce((a, b) => a + b) / size.toDouble();
    final centered = x - ValueVector(List.generate(size, (_) => mean));
    final variance = centered.values.map((v) => v * v).reduce((a, b) => a + b) / size.toDouble();
    final std = (variance + Value(eps)).sqrt();

    final normalized = ValueVector([
      for (int i = 0; i < size; i++) (x.values[i] - mean) / std
    ]);

    return ValueVector([
      for (int i = 0; i < size; i++) normalized.values[i] * gamma[i] + beta[i]
    ]);
  }
}
