// file: object_detection_head.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A simple object detection head for a ViT backbone, predicting multiple objects.
///
/// This head takes the output features from the ViT backbone (e.g., the CLS token
/// output) and predicts a fixed number of `numQueries` bounding box coordinates
/// and class probabilities.
///
/// Note: This is a simplified approach. In architectures like DETR, a Transformer
/// Decoder processes learnable object queries to generate these predictions.
class ObjectDetectionHead extends Module {
  final int embedSize; // Input feature dimension from the backbone
  final int numClasses; // Number of object classes (e.g., COCO has 80)
  final int numQueries; // Fixed number of object predictions to output
  static int numBoxCoords = 4; // x_center, y_center, width, height

  // Linear layer for bounding box prediction for all queries
  final Layer bboxRegressionHead;
  // Linear layer for class prediction for all queries
  final Layer classPredictionHead;

  ObjectDetectionHead({
    required this.embedSize,
    required this.numClasses,
    required this.numQueries,
  })  : bboxRegressionHead =
            Layer.fromNeurons(embedSize, numQueries * numBoxCoords),
        classPredictionHead = Layer.fromNeurons(embedSize,
            numQueries * (numClasses + 1)); // +1 for background class

  /// Forward pass for the object detection head.
  ///
  /// Takes a single `ValueVector` representing the aggregated image feature
  /// (e.g., the CLS token output from the ViT backbone).
  ///
  /// Returns a Map containing:
  /// - 'boxes': List of `numQueries` ValueVectors, each of 4 bounding box coordinates
  /// - 'logits': List of `numQueries` ValueVectors, each of (numClasses + 1) class logits
  Map<String, List<ValueVector>> forward(ValueVector backboneFeature) {
    // Predict flattened bounding box coordinates for all queries
    final ValueVector rawBboxesFlat =
        bboxRegressionHead.forward(backboneFeature);

    // Predict flattened class logits for all queries
    final ValueVector classLogitsFlat =
        classPredictionHead.forward(backboneFeature);

    // Reshape flattened outputs into lists of ValueVectors for each query
    final List<ValueVector> predictedBoxes = [];
    for (int i = 0; i < numQueries; i++) {
      predictedBoxes.add(ValueVector(rawBboxesFlat.values
          .sublist(i * numBoxCoords, (i + 1) * numBoxCoords)));
    }

    final List<ValueVector> predictedLogits = [];
    for (int i = 0; i < numQueries; i++) {
      predictedLogits.add(ValueVector(classLogitsFlat.values
          .sublist(i * (numClasses + 1), (i + 1) * (numClasses + 1))));
    }

    return {
      'boxes': predictedBoxes,
      'logits': predictedLogits,
    };
  }

  @override
  List<Value> parameters() {
    return [
      ...bboxRegressionHead.parameters(),
      ...classPredictionHead.parameters(),
    ];
  }
}
