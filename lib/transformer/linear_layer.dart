import 'dart:math';
import '../nn/value_vector.dart';
import '../nn/value.dart';

class LinearLayer {
  final int inputSize;
  final int outputSize;
  late List<ValueVector> weights; // shape: outputSize x inputSize
  late ValueVector biases; // shape: outputSize

  LinearLayer(this.inputSize, this.outputSize) {
    final rand = Random();
    weights = List.generate(outputSize, (_) {
      return ValueVector(List.generate(inputSize, (_) {
        return Value((rand.nextDouble() - 0.5) * 2 / sqrt(inputSize));
      }));
    });
    biases = ValueVector(List.generate(outputSize, (_) => Value(0.0)));
  }

  ValueVector forward(ValueVector input) {
    final outputs = <Value>[];
    for (var i = 0; i < outputSize; i++) {
      final weightedSum = weights[i].dot(input) + biases.values[i];
      outputs.add(weightedSum);
    }
    return ValueVector(outputs);
  }

  @override
  String toString() {
    return 'Linear($inputSize → $outputSize)';
  }
}
