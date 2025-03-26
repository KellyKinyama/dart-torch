import 'dart:math' as math;

import 'layer.dart';
import 'module.dart';
import 'neuron.dart';
import 'value.dart';
import 'value_vector.dart';





class MultiLayerPerceptron extends Module {
  // late int size;
  // late int nin;
  // late int nout;
  Layer inputLayer = Layer.fromNeurons(2, 2);
  Layer hiddenLayer = Layer.fromNeurons(2, 1);
  ValueVector? activatedValues;

  MultiLayerPerceptron() {
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
    // print("Output: $out2");
    return out2;
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    // for (Neuron neuron in inputLayer) {
    params.addAll(inputLayer.parameters());
    if (activatedValues != null) params.addAll(activatedValues!.values);
    params.addAll(hiddenLayer.parameters());
    // }
    // TODO: implement parameters
    return params;
  }
}

