import 'dart:math' as math;

import 'module.dart';
import 'value.dart';
import 'value_vector.dart';

class Neuron extends Module {
  ValueVector w;
  Value? b;
  bool nonlin = true;

  Neuron(this.w, {this.b, this.nonlin = true}) : super();

  factory Neuron.fromWeights(int nin, {bool nonlin = true}) {
    // He Initialization for weights
    // For ReLU, standard deviation is sqrt(2 / nin)
    final double stdDev = math.sqrt(2 / nin);

    final w = List<Value>.generate(
        nin,
        (int index) => Value(
            // Sample from a normal distribution with mean 0 and stdDev
            // Dart's math.Random().nextDouble() gives a uniform distribution between 0.0 and 1.0.
            // A common way to approximate a normal distribution from a uniform one
            // for initialization is using Box-Muller transform or simpler approximations.
            // For practical purposes in this context, multiplying by stdDev
            // with a centered random value is a common heuristic if a true
            // normal distribution sampler isn't readily available or needed for simplicity.
            // A more rigorous approach would involve a proper Box-Muller implementation.
            (math.Random().nextDouble() * 2 - 1) *
                stdDev // centered random between -stdDev and +stdDev
            ),
        growable: false);

    Value b = Value(0.0); // Often good to initialize biases to 0
    return Neuron(ValueVector(w), b: b, nonlin: true);
  }

  Value forward(ValueVector x) {
    final matMul = w.dot(x);

    // Apply nonlinearity if nonlin is true
    Value output = b == null ? matMul : matMul + b!;
    if (nonlin) {
      // Assuming you want to apply ReLU by default if nonlin is true
      output = output.relu(); // Or other activation like tanh, sigmoid, etc.
    }
    return output;
  }

  @override
  List<Value> parameters() {
    // TODO: implement parameters
    return [
      ...w.values,
      if (b != null) b!
    ]; // Only include bias if it's not null
  }
}
