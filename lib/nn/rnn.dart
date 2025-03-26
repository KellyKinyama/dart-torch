import 'dart:math' as math;
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'matrix2d.dart';

class RNN {
  late Matrix2d Wx; // Input to hidden
  late Matrix2d Wh; // Hidden to hidden
  late Matrix2d Wy; // Hidden to output
  late ValueVector bh; // Hidden bias
  late ValueVector by; // Output bias
  int inputSize;
  int hiddenSize;
  int outputSize;

  RNN(this.inputSize, this.hiddenSize, this.outputSize) {
    Wx = Matrix2d(hiddenSize, inputSize);
    Wh = Matrix2d(hiddenSize, hiddenSize);
    Wy = Matrix2d(outputSize, hiddenSize);
    bh = ValueVector(List.generate(hiddenSize, (_) => Value(0)));
    by = ValueVector(List.generate(outputSize, (_) => Value(0)));
  }

  ValueVector activate(ValueVector x) {
    // Simple activation function (tanh)
    return ValueVector(x.values.map((v) => v.tanh()).toList());
  }

  (ValueVector, ValueVector) step(ValueVector input, ValueVector prevHidden) {
    Matrix2d xMat = Matrix2d(inputSize, 1, input);
    Matrix2d hMat = Matrix2d(hiddenSize, 1, prevHidden);
    
    ValueVector newHidden = activate((Wx * xMat + Wh * hMat).data! + bh);
    ValueVector output = (Wy * Matrix2d(hiddenSize, 1, newHidden)).data! + by;
    return (newHidden, output);
  }

  List<ValueVector> forward(List<ValueVector> inputs) {
    ValueVector hidden = ValueVector(List.generate(hiddenSize, (_) => Value(0)));
    List<ValueVector> outputs = [];
    for (ValueVector input in inputs) {
      var (newHidden, output) = step(input, hidden);
      hidden = newHidden;
      outputs.add(output);
    }
    return outputs;
  }
}

void main() {
  final rnn = RNN(3, 5, 2);
  List<ValueVector> inputs = [
    ValueVector([Value(0.1), Value(0.2), Value(0.3)]),
    ValueVector([Value(0.4), Value(0.5), Value(0.6)]),
    ValueVector([Value(0.7), Value(0.8), Value(0.9)])
  ];

  List<ValueVector> outputs = rnn.forward(inputs);
  for (var output in outputs) {
    print("Output: ${output.values.map((v) => v.data).toList()}");
  }
}
