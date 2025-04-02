// void main() {
//   final input = ValueVector([Value(1), Value(0)]);
//   final nn = MultiLayerPerceptron();
//   final out = nn.forward(input);
//   print("Output: ${out}");
// }

// void main() {
//   final model = MultiLayerPerceptron(); // 4 inputs → 2 outputs

//   final inputs = [
//     ValueVector([Value(0.0), Value(0.0)]),
//     ValueVector([Value(0.0), Value(1.0)]),
//     ValueVector([Value(1.0), Value(0.0)]),
//     ValueVector([Value(1.0), Value(1.0)]),
//   ];

//   final targets = [
//     ValueVector([Value(0.0)]),
//     ValueVector([Value(1.0)]),
//     ValueVector([Value(1.0)]),
//     ValueVector([Value(0.0)]),
//   ];

//   const epochs = 10000;
//   const lr = 0.01;

//   for (int epoch = 0; epoch < epochs; epoch++) {
//     final losses = <Value>[];

//     // Reset gradients
//     for (var p in model.parameters()) {
//       p.grad = 0;
//     }

//     // Compute loss for all samples
//     for (int i = 0; i < inputs.length; i++) {
//       final yPred = model.forward(inputs[i]);
//       final yTrue = targets[i];
//       final diff = yPred - yTrue;
//       final squared = diff.squared();
//       final sampleLoss = squared.mean();
//       losses.add(sampleLoss);
//     }

//     final totalLoss = losses.reduce((a, b) => a + b);
//     // final avgLoss = totalLoss * (1.0 / inputs.length);
//     totalLoss.backward();

//     // Gradient descent
//     for (var p in model.parameters()) {
//       p.setData(p.data - lr * p.grad);
//     }

//     print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(4)}");
//   }

//   for (var input in inputs) {
//     // Reset gradients
//     for (var p in model.parameters()) {
//       p.grad = 0;
//     }
//     print("Input: ${input}");
//     print("Output: ${model.forward(input)}");
//   }
// }

import '../layer.dart';
import '../module.dart';
import '../value.dart';
import '../value_vector.dart';

class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  num lr;

  Layer inputLayer = Layer.fromNeurons(2, 2);
  Layer hiddenLayer = Layer.fromNeurons(2, 1);
  ValueVector? activatedValues;
  ValueVector? activatedOut;

  MultiLayerPerceptron(this.lr) {
    // print("input Layer length: ${inputLayer.neurons.length}");
    // print("hidden Layer length: ${hiddenLayer.neurons.length}");
  }

  // List<int> topology;
  // MultiLayerPerceptron(this.topology) {
  //   layers = List<Layer>.generate(topology.length, (int slot) {
  //     return Layer.fromNeurons(topology[slot], topology[slot + 1]);
  //   }, growable: false);
  // }

  ValueVector forward(ValueVector x) {
    final out = inputLayer.forward(x);
    final activated = out.sigmoid();
    activatedValues = activated;
    // print("input Layer neurons length: ${out.values.length}");
    final out2 = hiddenLayer.forward(activated);

    final activatedOut2 = out2.sigmoid();
    activatedOut = activatedOut2;
    // print("Output: $out2");
    return activatedOut2;
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    // for (Neuron neuron in inputLayer) {
    params.addAll(inputLayer.parameters());
    if (activatedValues != null) params.addAll(activatedValues!.values);
    params.addAll(hiddenLayer.parameters());
    if (activatedOut != null) params.addAll(activatedOut!.values);
    // }
    // TODO: implement parameters
    return params;
  }

  @override
  void zeroGrad() {
    // Reset gradients
    for (var p in parameters()) {
      p.grad = 0;
    }
  }

  void updateWeights() {
    // Gradient descent
    for (var p in parameters()) {
      p.setData(p.data - lr * p.grad);
    }
  }
}

void main() {
  const lr = 0.02;
  final model = MultiLayerPerceptron(lr); // 4 inputs → 2 outputs

  final inputs = [
    ValueVector([Value(0.0), Value(0.0)]),
    ValueVector([Value(0.0), Value(1.0)]),
    ValueVector([Value(1.0), Value(0.0)]),
    ValueVector([Value(1.0), Value(1.0)]),
  ];

  final targets = [
    ValueVector([Value(0.0)]),
    ValueVector([Value(1.0)]),
    ValueVector([Value(1.0)]),
    ValueVector([Value(0.0)]),
  ];

  const epochs = 100000;

  for (int epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Reset gradients
    model.zeroGrad();

    // Compute loss for all samples
    for (int i = 0; i < inputs.length; i++) {
      final yPred = model.forward(inputs[i]);
      final yTrue = targets[i];
      final diff = yPred - yTrue;
      final squared = diff.squared();
      final sampleLoss = squared.mean();
      losses.add(sampleLoss);
    }

    final totalLoss = losses.reduce((a, b) => a + b);
    // final avgLoss = totalLoss * (1.0 / inputs.length);
    // avgLoss.backward();
    totalLoss.backward();

    // Gradient descent
    model.updateWeights();

    if (epoch % 2000 == 0) {
      print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(4)}");
    }
  }

  for (var input in inputs) {
    // Reset gradients
    for (var p in model.parameters()) {
      p.grad = 0;
    }
    print("Input: ${input}");
    print("Output: ${model.forward(input)}");
    print("");
  }
}
