// file: object_detector.dart

import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'vit_backbone.dart'; // Your modified ViT backbone
import 'object_detection_head.dart'; // Your new detection head

/// A complete Vision Transformer-based Object Detector.
///
/// This model combines a ViT backbone for feature extraction with a simple
/// detection head for predicting bounding boxes and class labels.
///
/// Note: This is a highly simplified architecture for demonstration.
/// Real-world object detectors like DETR, Faster R-CNN, YOLO are significantly
/// more complex, involving multi-scale features, sophisticated heads,
/// and specialized loss functions.
class ViTObjectDetector extends Module {
  final ViTBackbone backbone;
  final ObjectDetectionHead detectionHead;

  ViTObjectDetector({
    required int imageSize,
    required int patchSize,
    required int numChannels,
    required int embedSize,
    required int numLayers,
    required int numHeads,
    required int numClasses, // Number of object classes (excluding background)
  })  : backbone = ViTBackbone(
            imageSize: imageSize,
            patchSize: patchSize,
            numChannels: numChannels,
            embedSize: embedSize,
            numLayers: numLayers,
            numHeads: numHeads),
        detectionHead = ObjectDetectionHead(
            embedSize: embedSize, numClasses: numClasses);

  /// Forward pass for the object detector.
  ///
  /// Takes a flattened list of image pixel data.
  /// Returns a Map containing object predictions.
  ///
  /// The `encodedFeatures` from the backbone will contain:
  /// [CLS_token_embedding, patch_embedding_1, patch_embedding_2, ...]
  /// For this simple head, we'll use the CLS token's output.
  Map<String, ValueVector> forward(List<double> imageData) {
    // Get contextualized features from the ViT backbone
    final List<ValueVector> encodedFeatures = backbone.forward(imageData);

    // For this simple detection head, we'll use the CLS token's output
    // as the global image representation for prediction.
    final ValueVector clsFeature = encodedFeatures[0];

    // Pass the CLS feature to the detection head
    final Map<String, ValueVector> predictions =
        detectionHead.forward(clsFeature);

    return predictions;
  }

  @override
  List<Value> parameters() {
    return [
      ...backbone.parameters(),
      ...detectionHead.parameters(),
    ];
  }
}