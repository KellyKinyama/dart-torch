import 'module.dart';
import 'neuron.dart';
import 'value.dart';
import 'value_vector.dart';

class Layer extends Module {
  // int nin;
  // int nout;
  List<Neuron> neurons;
  Layer(this.neurons);

  factory Layer.fromNeurons(int nin, int nout) {
    final neurons = List<Neuron>.generate(
        nout, (int index) => Neuron.fromWeights(nin),
        growable: false);
    return Layer(neurons);
  }

  ValueVector forward(ValueVector x) {
    final out = List.generate(neurons.length, (int index) {
      // print("weights length: ${neurons[index].w.values.length}");
      // print("input length:   ${x.values.length}");
      return neurons[index].forward(x);
    });
    return ValueVector(out);
  }

  @override
  List<Value> parameters() {
    final List<Value> params = [];
    for (Neuron neuron in neurons) {
      params.addAll(neuron.parameters());
    }
    // TODO: implement parameters
    return params;
  }
}