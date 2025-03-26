import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'matrix2d.dart';

class Flatten {
  /// Flattens multi-channel 2D feature maps into a 1D vector
  ValueVector forward(List<Matrix2d> input) {
    List<Value> flatValues = [];

    // Iterate through all channels and flatten each matrix
    for (var channel in input) {
      flatValues.addAll(channel.data!.values);
    }

    return ValueVector(flatValues);
  }
}

void main() {
  // Example feature maps (3-channel, each 3x3)
  List<Matrix2d> featureMaps = [
    Matrix2d(3, 3), // Feature map 1
    Matrix2d(3, 3), // Feature map 2
    Matrix2d(3, 3), // Feature map 3
  ];

  // Create Flatten layer
  Flatten flatten = Flatten();

  // Apply flattening
  ValueVector flattenedOutput = flatten.forward(featureMaps);

  print("Flattened Output Size: ${flattenedOutput.values.length}");
}
