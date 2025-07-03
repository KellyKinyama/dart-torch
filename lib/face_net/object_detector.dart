// file: object_detector.dart

import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../vit/vit_backbone.dart'; // Your modified ViT backbone
import 'object_detection_head.dart'; // Your new detection head

/// A complete Vision Transformer-based model for Face Detection and Recognition.
///
/// This model combines a ViT backbone for feature extraction with a detection
/// head that predicts bounding boxes, class labels (identities + background),
/// and a discriminative embedding for each detected face.
class ViTObjectDetector extends Module {
  // Renamed conceptually to indicate broader use
  final ViTBackbone backbone;
  final ObjectDetectionHead detectionHead;
  final int numQueries; // Fixed number of object predictions
  final int embeddingDim; // Dimension of the face embedding

  ViTObjectDetector({
    required int imageSize,
    required int patchSize,
    required int numChannels,
    required int embedSize,
    required int numLayers,
    required int numHeads,
    required int
        numClasses, // Number of object classes (identities + background)
    required this.numQueries,
    required this.embeddingDim, // New parameter
  })  : backbone = ViTBackbone(
            imageSize: imageSize,
            patchSize: patchSize,
            numChannels: numChannels,
            embedSize: embedSize,
            numLayers: numLayers,
            numHeads: numHeads),
        detectionHead = ObjectDetectionHead(
            embedSize: embedSize,
            numClasses: numClasses,
            numQueries: numQueries,
            embeddingDim: embeddingDim); // Pass new parameter

  /// Forward pass for the face detection and recognition model.
  ///
  /// Takes a flattened list of image pixel data.
  /// Returns a Map containing lists of object predictions (boxes, logits, embeddings).
  Map<String, List<ValueVector>> forward(List<double> imageData) {
    // Get contextualized features from the ViT backbone
    final List<ValueVector> encodedFeatures = backbone.forward(imageData);

    // For this simple detection head, we'll use the CLS token's output
    // as the global image representation for prediction.
    final ValueVector clsFeature = encodedFeatures[0];

    // Pass the CLS feature to the detection head, which will produce multiple predictions
    final Map<String, List<ValueVector>> predictions =
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
