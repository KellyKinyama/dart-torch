import 'dart:math';
import 'dart:typed_data';

import '../../nn/layer.dart';
import '../../nn/module.dart';
import '../../nn/value.dart';
import '../../nn/value_vector.dart';

class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  num lr;

  Layer inputLayer = Layer.fromNeurons(784, 128);
  Layer hiddenLayer = Layer.fromNeurons(128, 10);
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

void trainModel(
    MultiLayerPerceptron model, List<Float32List> images, List<int> labels,
    {int epochs = 5, int batchSize = 32}) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Reset gradients
    // print("zeroing gradients");
    model.zeroGrad();
    int samplesLength = 0;
    for (int i = 0; i < images.length; i++) {
      final rand = Random().nextInt(images.length - 1);
      final input = images[rand];
      final target = List<double>.filled(10, 0.0);
      target[labels[rand]] = 1.0; // One-hot encoding

      // Compute loss for all samples
      // print("Predicting");
      final yPred = model.forward(ValueVector.fromFloat32List(input));
      final yTrue = ValueVector.fromDoubleList(target);
      final diff = yPred - yTrue;
      final squared = diff.squared();
      final sampleLoss = squared.mean();

      // print("Sample loss: $sampleLoss");
      losses.add(sampleLoss);
      if (losses.length > batchSize) break;
      samplesLength++;
    }

    final totalLoss = losses.reduce((a, b) => a + b);
    final avgLoss = totalLoss * (1.0 / samplesLength);
    // print("Average loss: $avgLoss");

    // print("Performing back propagation");
    avgLoss.backward();

    // Gradient descent
    // print("Weightd update");
    model.updateWeights();

    print('Epoch $epoch complete. Average Loss: $avgLoss');
  }
}
