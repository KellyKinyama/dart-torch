// file: object_detection_head.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A simple object detection head for a ViT backbone.
///
/// This head takes the output features from the ViT backbone (e.g., the CLS token
/// output or aggregated patch features) and predicts bounding box coordinates
/// and class probabilities for a *single* object.
///
/// For multi-object detection, a more advanced architecture like DETR's
/// Transformer Decoder or R-CNN style heads would be required.
class ObjectDetectionHead extends Module {
  final int embedSize; // Input feature dimension from the backbone
  final int numClasses; // Number of object classes (e.g., COCO has 80)
  static const int numBoxCoords = 4; // x_center, y_center, width, height

  // Linear layer for bounding box prediction
  final Layer bboxRegressionHead;
  // Linear layer for class prediction
  final Layer classPredictionHead;

  ObjectDetectionHead({
    required this.embedSize,
    required this.numClasses,
  })  : bboxRegressionHead = Layer.fromNeurons(embedSize, numBoxCoords),
        classPredictionHead = Layer.fromNeurons(embedSize, numClasses + 1); // +1 for background class

  /// Forward pass for the object detection head.
  ///
  /// Takes a single `ValueVector` representing the aggregated image feature
  /// (e.g., the CLS token output from the ViT backbone).
  ///
  /// Returns a Map containing:
  /// - 'boxes': ValueVector of 4 bounding box coordinates
  /// - 'logits': ValueVector of class logits (including background)
  Map<String, ValueVector> forward(ValueVector backboneFeature) {
    // Predict bounding box coordinates
    final ValueVector rawBbox = bboxRegressionHead.forward(backboneFeature);

    // Predict class logits
    final ValueVector classLogits = classPredictionHead.forward(backboneFeature);

    return {
      'boxes': rawBbox,
      'logits': classLogits,
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
