import 'linear_layer.dart';
import '../nn/value_vector.dart';

class ClassifierHead {
  final LinearLayer fc;

  ClassifierHead(int inputDim, int numClasses)
      : fc = LinearLayer(inputDim, numClasses);

  ValueVector forward(ValueVector x) => fc.forward(x);
}
