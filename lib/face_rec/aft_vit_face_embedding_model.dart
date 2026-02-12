// file: vit_face_embedding_model.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer/aft_transformer_encoder.dart'; // Assuming this path is correct for your TransformerEncoder

/// A Vision Transformer (ViT) model adapted for generating face embeddings.
///
/// This model processes an image by dividing it into patches, linearly
/// embedding them, adding positional information, and feeding them through
/// a Transformer Encoder. The output of a special [CLS] token is then
/// used as the discriminative face embedding.
class ViTFaceEmbeddingModel extends Module {
  final int imageSize; // e.g., 224 (assuming square images)
  final int patchSize; // e.g., 16x16
  final int numChannels; // e.g., 3 for RGB, 1 for grayscale
  final int
      embedSize; // Dimension of patch embeddings and Transformer (this will be the embedding size)
  final int numLayers; // Number of encoder layers
  final int numHeads; // Number of attention heads

  // The linear projection layer for patches
  final Layer patchProjection;

  // The special learnable [CLS] token
  final ValueVector clsToken;

  // Positional embeddings (learned)
  // +1 because we include the CLS token in the sequence
  final List<ValueVector> positionEmbeddings;

  // The main Transformer Encoder backbone
  final TransformerEncoder transformerEncoder;

  ViTFaceEmbeddingModel({
    required this.imageSize,
    required this.patchSize,
    this.numChannels = 3,
    required this.embedSize, // This will be the output embedding size
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
    final patchPixelCount = patchSize * patchSize * numChannels;

    final List<ValueVector> patchEmbeddings = [];

    for (int patchY = 0; patchY < numPatchesPerRow; patchY++) {
      for (int patchX = 0; patchX < numPatchesPerRow; patchX++) {
        final List<double> currentPatchPixels = [];

        for (int c = 0; c < numChannels; c++) {
          for (int y = 0; y < patchSize; y++) {
            for (int x = 0; x < patchSize; x++) {
              final originalPixelX = patchX * patchSize + x;
              final originalPixelY = patchY * patchSize + y;
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
              "Extracted patch pixel count (${currentPatchPixels.length}) does not match expected ($patchPixelCount).");
        }

        final patchVector = ValueVector.fromDoubleList(currentPatchPixels);
        patchEmbeddings.add(patchProjection.forward(patchVector));
      }
    }
    return patchEmbeddings;
  }

  /// The forward pass for the ViT Face Embedding Model.
  ///
  /// Takes a flattened list of image pixel data.
  /// Returns the face embedding (ValueVector) derived from the CLS token.
  ValueVector forward(List<double> imageData) {
    // 1. Create patch embeddings
    final patchEmbeddings = _createPatchesAndEmbeddings(imageData);

    // 2. Prepend the learnable [CLS] token
    // Create a new ValueVector for clsToken to ensure it's part of the current graph.
    // Otherwise, its gradient won't be calculated if it's the same object every time.
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

    // 5. Take the output corresponding to the [CLS] token (first element)
    // This CLS token output serves as the face embedding.
    final ValueVector clsOutput = encodedFeatures[0];

    return clsOutput; // Return the embedding
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
