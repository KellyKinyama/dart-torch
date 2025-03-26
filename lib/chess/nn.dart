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

import '../nn/layer.dart';
import '../nn/module.dart';
import '../nn/value_vector.dart';
import '../nn/value.dart';

class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  num lr;

  Layer inputLayer = Layer.fromNeurons(129, 129);
  Layer hiddenLayer = Layer.fromNeurons(129, 129);

  // Layer outLayer = Layer.fromNeurons(129, 2048);

  Layer outLayer = Layer.fromNeurons(129, 4096);
  ValueVector? activatedValues;
  ValueVector? activatedOut;
  ValueVector? activatedOut3;

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

  Future<ValueVector> forward(ValueVector x) async {
    final out = inputLayer.forward(x);
    final activated = out.sigmoid();
    activatedValues = activated;
    // print("input Layer neurons length: ${out.values.length}");
    final out2 = hiddenLayer.forward(activated);

    final out3 = out2.sigmoid();
    final out4 = outLayer.forward(out3);

    final activatedOut = out4.sigmoid();
    activatedOut3 = activatedOut;
    // print("Output: $out2");
    return activatedOut;
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    // for (Neuron neuron in inputLayer) {
    params.addAll(inputLayer.parameters());
    if (activatedValues != null) params.addAll(activatedValues!.values);
    params.addAll(hiddenLayer.parameters());
    params.addAll(outLayer.parameters());
    if (activatedOut3 != null) params.addAll(activatedOut3!.values);
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

// void main() {
//   const lr = 0.02;
//   final model = MultiLayerPerceptron(lr); // 4 inputs → 2 outputs

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

//   const epochs = 100000;

//   for (int epoch = 0; epoch < epochs; epoch++) {
//     final losses = <Value>[];

//     // Reset gradients
//     model.zeroGrad();

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
//     final avgLoss = totalLoss * (1.0 / inputs.length);
//     avgLoss.backward();

//     // Gradient descent
//     model.updateWeights();

//     if (epoch % 2000 == 0) {
//       print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(4)}");
//     }
//   }

//   for (var input in inputs) {
//     // Reset gradients
//     for (var p in model.parameters()) {
//       p.grad = 0;
//     }
//     print("Input: ${input}");
//     print("Output: ${model.forward(input)}");
//     print("");
//   }
// }
