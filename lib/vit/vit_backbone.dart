// file: vit_backbone.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer/transformer_encoder2.dart'; // The encoder you already have

/// A Vision Transformer (ViT) backbone for extracting image features.
///
/// This model processes an image by dividing it into patches, linearly
/// embedding them, adding positional information, and feeding them through
/// a Transformer Encoder. It outputs the contextualized embeddings of all
/// patches (and optionally a CLS token) for downstream tasks like object detection.
class ViTBackbone extends Module {
  final int imageSize; // e.g., 224 (assuming square images)
  final int patchSize; // e.g., 16x16
  final int numChannels; // e.g., 3 for RGB, 1 for grayscale
  final int embedSize; // Dimension of patch embeddings and Transformer
  final int numLayers; // Number of encoder layers
  final int numHeads; // Number of attention heads

  // The linear projection layer for patches
  final Layer patchProjection;

  // The special learnable [CLS] token (optional for detection, but kept for consistency)
  final ValueVector clsToken;

  // Positional embeddings (learned)
  // +1 because we include the CLS token in the sequence
  final List<ValueVector> positionEmbeddings;

  // The main Transformer Encoder backbone
  final TransformerEncoder transformerEncoder;

  ViTBackbone({
    required this.imageSize,
    required this.patchSize,
    this.numChannels = 3,
    required this.embedSize,
    this.numLayers = 2, // Reduced for faster example execution
    this.numHeads = 4, // Reduced for faster example execution
  })  : assert(imageSize % patchSize == 0,
            "Image size must be divisible by patch size"),
        assert(embedSize % numHeads == 0,
            "Embed size must be divisible by numHeads"),
        // Patch embedding converts (patch_size * patch_size * num_channels) into embedSize
        patchProjection =
            Layer.fromNeurons(patchSize * patchSize * numChannels, embedSize),
        // Initialize CLS token as a learnable vector
        clsToken = ValueVector.fromDoubleList(List.generate(
            embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01)),
        // Calculate the number of patches along one side
        // Example: 224 / 16 = 14 patches per side -> 14 * 14 = 196 patches total
        // Plus 1 for the [CLS] token: (num_patches + 1) positions
        positionEmbeddings = List.generate(
            (imageSize ~/ patchSize) * (imageSize ~/ patchSize) + 1,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // The TransformerEncoder is used as the backbone
        // Its blockSize must match the sequence length (patches + CLS token)
        transformerEncoder = TransformerEncoder(
          vocabSize: 0, // Not used, as embeddings are provided directly
          embedSize: embedSize,
          blockSize: (imageSize ~/ patchSize) * (imageSize ~/ patchSize) + 1,
          numLayers: numLayers,
          numHeads: numHeads,
        );

  /// Converts a flattened image (List<double>) into a list of embedded patch vectors.
  ///
  /// This method conceptually performs patching and linear projection.
  /// For real images, you'd need actual image processing/tensor manipulation
  /// to correctly extract patch pixel values.
  List<ValueVector> _createPatchesAndEmbeddings(List<double> imageData) {
    final numPixels = imageSize * imageSize * numChannels;
    if (imageData.length != numPixels) {
      throw ArgumentError(
          "Image data length (${imageData.length}) does not match expected size ($numPixels).");
    }

    final numPatchesPerRow = imageSize ~/ patchSize;
    final numTotalPatches = numPatchesPerRow * numPatchesPerRow;
    final patchPixelCount = patchSize * patchSize * numChannels;

    final List<ValueVector> patchEmbeddings = [];

    // Conceptually extract patches row by row, then within each row.
    for (int patchY = 0; patchY < numPatchesPerRow; patchY++) {
      for (int patchX = 0; patchX < numPatchesPerRow; patchX++) {
        final List<double> currentPatchPixels = [];

        for (int c = 0; c < numChannels; c++) {
          for (int y = 0; y < patchSize; y++) {
            for (int x = 0; x < patchSize; x++) {
              final originalPixelX = patchX * patchSize + x;
              final originalPixelY = patchY * patchSize + y;
              // Assuming channel-last flattening (HWC) then flattened to 1D
              final pixelFlatIndex =
                  (originalPixelY * imageSize + originalPixelX) * numChannels +
                      c;
              if (pixelFlatIndex >= 0 && pixelFlatIndex < imageData.length) {
                currentPatchPixels.add(imageData[pixelFlatIndex]);
              } else {
                throw StateError(
                    "Calculated pixel index out of bounds. Check image data flattening.");
              }
            }
          }
        }

        if (currentPatchPixels.length != patchPixelCount) {
          throw StateError(
              "Extracted patch pixel count (${currentPatchPixels.length}) does not match expected (${patchPixelCount}).");
        }

        final patchVector = ValueVector.fromDoubleList(currentPatchPixels);
        patchEmbeddings.add(patchProjection.forward(patchVector));
      }
    }
    return patchEmbeddings;
  }

  /// The forward pass for the ViT Backbone.
  ///
  /// Takes a flattened list of image pixel data.
  /// Returns a list of contextualized `ValueVector`s (CLS token + patch embeddings).
  List<ValueVector> forward(List<double> imageData) {
    // 1. Create patch embeddings
    final patchEmbeddings = _createPatchesAndEmbeddings(imageData);

    // 2. Prepend the learnable [CLS] token
    final currentClsToken = ValueVector(List.generate(
        embedSize,
        (i) =>
            Value(clsToken.values[i].data, {clsToken.values[i]}, 'cls_copy')));
    final sequence = [currentClsToken, ...patchEmbeddings];

    // 3. Add positional embeddings
    final sequenceWithPositionalEmbeddings =
        List.generate(sequence.length, (i) {
      return sequence[i] + positionEmbeddings[i];
    });

    // 4. Pass the sequence through the Transformer Encoder
    final encodedFeatures =
        transformerEncoder.forwardEmbeddings(sequenceWithPositionalEmbeddings);

    // Return all encoded features (CLS token + patch embeddings)
    return encodedFeatures;
  }

  @override
  List<Value> parameters() {
    return [
      ...patchProjection.parameters(),
      ...clsToken.values, // CLS token itself is a parameter
      ...positionEmbeddings.expand((vec) => vec.values),
      ...transformerEncoder.parameters(),
    ];
  }
}
