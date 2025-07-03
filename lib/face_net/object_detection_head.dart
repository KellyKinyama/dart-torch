// file: object_detection_head.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A simple object detection head for a ViT backbone, predicting multiple objects.
///
/// This head takes the output features from the ViT backbone (e.g., the CLS token
/// output) and predicts a fixed number of `numQueries` bounding box coordinates,
/// class probabilities, and a face embedding for each.
class ObjectDetectionHead extends Module {
  final int embedSize; // Input feature dimension from the backbone
  final int numClasses; // Number of object classes (e.g., 5 identities)
  final int numQueries; // Fixed number of object predictions to output
  final int embeddingDim; // Dimension of the face embedding
  static int numBoxCoords = 4; // x_center, y_center, width, height

  // Linear layer for bounding box prediction for all queries
  final Layer bboxRegressionHead;
  // Linear layer for class prediction for all queries
  final Layer classPredictionHead;
  // NEW: Linear layer for face embedding prediction for all queries
  final Layer faceEmbeddingHead;

  ObjectDetectionHead({
    required this.embedSize,
    required this.numClasses,
    required this.numQueries,
    required this.embeddingDim, // New parameter for embedding dimension
  })  : bboxRegressionHead =
            Layer.fromNeurons(embedSize, numQueries * numBoxCoords),
        classPredictionHead = Layer.fromNeurons(embedSize,
            numQueries * (numClasses + 1)), // +1 for background class
        faceEmbeddingHead = Layer.fromNeurons(embedSize,
            numQueries * embeddingDim); // Output embedding for each query

  /// Forward pass for the object detection head.
  ///
  /// Takes a single `ValueVector` representing the aggregated image feature
  /// (e.g., the CLS token output from the ViT backbone).
  ///
  /// Returns a Map containing:
  /// - 'boxes': List of `numQueries` ValueVectors, each of 4 bounding box coordinates
  /// - 'logits': List of `numQueries` ValueVectors, each of (numClasses + 1) class logits
  /// - 'embeddings': List of `numQueries` ValueVectors, each of `embeddingDim` dimensions
  Map<String, List<ValueVector>> forward(ValueVector backboneFeature) {
    // Predict flattened bounding box coordinates for all queries
    final ValueVector rawBboxesFlat =
        bboxRegressionHead.forward(backboneFeature);

    // Predict flattened class logits for all queries
    final ValueVector classLogitsFlat =
        classPredictionHead.forward(backboneFeature);

    // NEW: Predict flattened face embeddings for all queries
    final ValueVector faceEmbeddingsFlat =
        faceEmbeddingHead.forward(backboneFeature);

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

    // NEW: Reshape flattened embeddings
    final List<ValueVector> predictedEmbeddings = [];
    for (int i = 0; i < numQueries; i++) {
      predictedEmbeddings.add(ValueVector(faceEmbeddingsFlat.values
          .sublist(i * embeddingDim, (i + 1) * embeddingDim)));
    }

    return {
      'boxes': predictedBoxes,
      'logits': predictedLogits,
      'embeddings': predictedEmbeddings, // Add new output
    };
  }

  @override
  List<Value> parameters() {
    return [
      ...bboxRegressionHead.parameters(),
      ...classPredictionHead.parameters(),
      ...faceEmbeddingHead.parameters(), // Add new head's parameters
    ];
  }
}
