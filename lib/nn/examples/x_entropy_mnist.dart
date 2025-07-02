import 'dart:typed_data';

import '../layer.dart';
import '../module.dart';
import '../value.dart';
import '../value_vector.dart';

class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  num lr;

  Layer inputLayer = Layer.fromNeurons(784, 200);
  Layer hiddenLayer = Layer.fromNeurons(200, 10);
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

    final activatedOut2 = out2.softmax();
    activatedOut = activatedOut2;
    // print("Output: $out2");
    return activatedOut2;
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    // for (Neuron neuron in inputLayer) {
    params.addAll(inputLayer.parameters());
    // if (activatedValues != null) {
    //   params.addAll(activatedValues!.values);
    // }
    params.addAll(hiddenLayer.parameters());
    // if (activatedOut != null) {
    //   params.addAll(activatedOut!.values);
    // }
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
  const lr = 0.008;
  final model = MultiLayerPerceptron(lr); // 4 inputs → 2 outputs
  print("Image bytes legnt: ${imgBytes.length}"); // 784 bytes
  final inputs = [
    ValueVector.fromDoubleList(imgBytes
        .map((e) => e / 255.0)
        .toList()), // Correct casting might be needed
    ValueVector.fromDoubleList(imgBytes.map((e) => e / 255.0).toList()),
    ValueVector.fromDoubleList(imgBytes.map((e) => e / 255.0).toList())
  ];

  final targets = [
    ValueVector([
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(1.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0)
    ]),
    ValueVector([
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(1.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0)
    ]),
    ValueVector([
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(1.0),
      Value(0.0),
      Value(0.0),
      Value(0.0),
      Value(0.0)
    ])
  ];

  const epochs = 400;

  for (int epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Reset gradients

    // Compute loss for all samples
    for (int i = 0; i < inputs.length; i++) {
      final yPred = model.forward(inputs[i]);
      final yTrue = targets[i];

      // Use crossEntropy instead of squared error
      final sampleLoss = yPred
          .crossEntropy(yTrue); // Assuming yPred and yTrue are ValueVectors
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
    if (epoch % 4 == 0) {
      print("Epoch $epoch | Loss = ${totalLoss.data.toStringAsFixed(10)}");
    }
    model.zeroGrad();
  }

  for (var input in inputs) {
    // Reset gradients
    for (var p in model.parameters()) {
      p.grad = 0;
    }
    // print("Input: ${input}");
    print("Output: ${model.forward(input)}");
    print("");
  }
}
