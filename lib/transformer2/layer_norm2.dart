// file: layer_norm.dart

import 'dart:math'; // Epsilon may need math.sqrt
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// Applies Layer Normalization over a single input vector.
///
/// This layer normalizes the activations of the previous layer for each token
/// independently. It helps in stabilizing the learning process.
class LayerNorm extends Module {
  final Value gamma; // Learnable scale parameter
  final Value beta; // Learnable shift parameter
  final double epsilon;

  LayerNorm(int dim, {this.epsilon = 1e-5})
      // Initialize gamma to 1 and beta to 0
      : gamma = Value(1.0),
        beta = Value(0.0);

  /// Forward pass for Layer Normalization.
  ///
  /// Takes a vector `x` and returns the normalized vector.
  ValueVector forward(ValueVector x) {
    final mean = x.mean();

    // CORRECTED PART: Manually perform element-wise subtraction
    // This creates a new vector where each element is (x_i - mean).
    final x_minus_mean = ValueVector(x.values.map((v) => v - mean).toList());

    final variance = x_minus_mean.squared().mean();

    // Normalize x to have mean 0 and variance 1. Use the new vector here.
    final xHat = x_minus_mean / (variance + epsilon).sqrt();

    // Scale and shift with learnable parameters
    return (xHat * gamma) + beta;
  }

  @override
  List<Value> parameters() {
    return [gamma, beta];
  }
}
