// In a new file: patch_embedding.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer/layer_norm2.dart'; // Assuming layer_norm2.dart contains LayerNorm

/// Converts an image into a sequence of patch embeddings.
class PatchEmbedding extends Module {
  final int patchSize; // e.g., 4 (for 4x4 patches)
  final int inChannels; // e.g., 3 for RGB images
  final int embedDim; // Output embedding dimension for each patch
  final Layer projection;
  final LayerNorm norm;

  PatchEmbedding({
    required this.patchSize,
    required this.inChannels,
    required this.embedDim,
  })  : projection =
            Layer.fromNeurons(patchSize * patchSize * inChannels, embedDim),
        norm = LayerNorm(embedDim);

  /// Forward pass for PatchEmbedding.
  ///
  /// Takes a flat list of `ValueVector`s representing image patches.
  /// Each `ValueVector` is `(patchSize * patchSize * inChannels)` dimensions.
  /// Returns a list of `ValueVector`s, each of `embedDim` dimensions.
  ///
  /// In a full image pipeline, you'd reshape your image (H, W, C) into
  /// (num_patches, patchSize*patchSize*C) first.
  List<ValueVector> forward(List<ValueVector> imagePatches) {
    // Each imagePatch in the list is already flattened (patchSize*patchSize*inChannels)
    // Project each flattened patch to embedDim
    final projectedPatches =
        imagePatches.map((patch) => projection.forward(patch)).toList();

    // Apply layer normalization
    final normalizedPatches =
        projectedPatches.map((patch) => norm.forward(patch)).toList();

    return normalizedPatches;
  }

  @override
  List<Value> parameters() {
    return [...projection.parameters(), ...norm.parameters()];
  }
}
