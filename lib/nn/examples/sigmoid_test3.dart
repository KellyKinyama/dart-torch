import 'dart:typed_data';

import '../layer.dart';
import '../module.dart';
import '../value.dart';
import '../value_vector.dart';

const DOUBLE_PI = 22 / 7 * 2;

class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  num lr;

  Layer inputLayer = Layer.fromNeurons(3, 200);
  Layer hiddenLayer = Layer.fromNeurons(200, 5);
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
    // if (activatedValues != null)
    params.addAll(activatedValues!.values);
    params.addAll(hiddenLayer.parameters());
    // if (activatedOut != null)
    params.addAll(activatedOut!.values);
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
      p.setData(p.data - (lr * p.grad));
    }
  }
}

void main() {
  const lr = 0.08;
  final model = MultiLayerPerceptron(lr); // 4 inputs → 2 outputs
  // print("Image bytes legnt: ${imgBytes.length}"); // 784 bytes
  final inputs = [
    ValueVector([
      Value(2 / DOUBLE_PI),
      Value(1.8 / DOUBLE_PI),
      Value(1.7 / DOUBLE_PI),
    ]),
    ValueVector([
      Value(1.6 / DOUBLE_PI),
      Value(1.5 / DOUBLE_PI),
      Value(1.4 / DOUBLE_PI),
    ]),
    ValueVector([
      Value(1.3 / DOUBLE_PI),
      Value(1.2 / DOUBLE_PI),
      Value(1.1 / DOUBLE_PI),
    ])
  ];

  final targets = [
    ValueVector([
      Value(2 / DOUBLE_PI),
      Value(1.8 / DOUBLE_PI),
      Value(1.7 / DOUBLE_PI),
      Value(1.6 / DOUBLE_PI),
      Value(1.5 / DOUBLE_PI),
    ]),
    ValueVector([
      Value(1.4 / DOUBLE_PI),
      Value(1.3 / DOUBLE_PI),
      Value(1.2 / DOUBLE_PI),
      Value(1.1 / DOUBLE_PI),
      Value(1.0 / DOUBLE_PI),
    ]),
    ValueVector([
      Value(0.9 / DOUBLE_PI),
      Value(0.8 / DOUBLE_PI),
      Value(0.7 / DOUBLE_PI),
      Value(0.6 / DOUBLE_PI),
      Value(0.5 / DOUBLE_PI),
    ])
  ];
  print("Inputs: $inputs"); // 784 bytes
  const epochs = 100000;

  for (int epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Reset gradients

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

    // if (epoch % 4 == 0) {
    //   print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(4)}");
    // }
    if (epoch % 1000 == 0) {
      print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(10)}");
    }
    model.zeroGrad();
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
