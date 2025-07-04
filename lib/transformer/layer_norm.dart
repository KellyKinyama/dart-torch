// layer_norm.dart
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

class LayerNormalization extends Module {
  final int features;
  Value gamma; // Learnable scale parameter
  Value beta; // Learnable shift parameter
  final double epsilon;

  LayerNormalization(this.features, {this.epsilon = 1e-5})
      : gamma = Value(1.0), // Typically initialized to 1
        beta = Value(0.0), // Typically initialized to 0
        super();

  @override
  List<Value> parameters() {
    return [gamma, beta];
  }

  ValueVector forward(ValueVector input) {
    if (input.values.length != features) {
      throw ArgumentError(
          "Input vector size (${input.values.length}) must match features ($features)");
    }

    // Calculate mean
    Value sum = input.values.reduce((a, b) => a + b);
    Value mean = sum / Value(features.toDouble());

    // Calculate variance (using data, not Value objects for intermediate step to avoid graph explosion)
    // For proper backprop, these intermediate operations should also create Value objects
    // This is a simplification; a full implementation would build these into the computational graph.
    Value varianceSum = Value(0.0);
    for (Value val in input.values) {
      varianceSum += (val - mean).pow(2);
    }
    Value variance = varianceSum / Value(features.toDouble());

    // Normalized input
    List<Value> normalizedValues = [];
    for (Value val in input.values) {
      normalizedValues.add((val - mean) / (variance + Value(epsilon)).sqrt());
    }
    ValueVector normalizedInput = ValueVector(normalizedValues);

    // Apply learned scale and shift
    List<Value> outputValues = [];
    for (Value val in normalizedInput.values) {
      outputValues.add(gamma * val + beta);
    }
    return ValueVector(outputValues);
  }
}
