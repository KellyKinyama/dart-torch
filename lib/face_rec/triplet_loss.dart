// file: triplet_loss.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// Implements the Triplet Loss function.
///
/// Triplet Loss aims to ensure that the distance between an anchor embedding
/// and a positive embedding (same identity) is smaller than the distance
/// between the anchor embedding and a negative embedding (different identity),
/// by at least a specified margin.
class TripletLoss extends Module {
  final double margin;

  TripletLoss({required this.margin});

  /// Calculates the Euclidean distance between two ValueVectors.
  static Value euclideanDistance(ValueVector a, ValueVector b) {
    assert(a.values.length == b.values.length,
        "Vectors must have the same dimension.");
    Value sumSquaredDiff = Value(0.0);
    for (int i = 0; i < a.values.length; i++) {
      sumSquaredDiff += (a.values[i] - b.values[i]).pow(2);
    }
    // Return the square root of the sum of squared differences
    // For backpropagation, it's often more stable to work with squared Euclidean distance
    // or to ensure the gradient of sqrt is handled correctly. Here, we'll use sqrt for true distance.
    return sumSquaredDiff.sqrt();
  }

  /// Computes the Triplet Loss.
  ///
  /// [anchor]: The embedding of the anchor image.
  /// [positive]: The embedding of the positive image (same identity as anchor).
  /// [negative]: The embedding of the negative image (different identity from anchor).
  ///
  /// Returns a Value representing the computed triplet loss.
  Value forward(
      ValueVector anchor, ValueVector positive, ValueVector negative) {
    // Calculate distances
    final Value distAP =
        euclideanDistance(anchor, positive); // Distance Anchor-Positive
    final Value distAN =
        euclideanDistance(anchor, negative); // Distance Anchor-Negative

    // Compute the loss: max(0, dist(A, P) - dist(A, N) + margin)
    final Value loss = (distAP - distAN + Value(margin)).relu();

    return loss;
  }

  @override
  List<Value> parameters() {
    // TripletLoss itself does not have learnable parameters,
    // but it inherits from Module for consistency if needed for a larger framework.
    return [];
  }
}
