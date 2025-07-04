// In a new file: patch_merging.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer/layer_norm2.dart'; // Assuming layer_norm2.dart contains LayerNorm

/// Merges 2x2 neighboring patches to reduce spatial resolution and increase feature dimension.
class PatchMerging extends Module {
  final int inDim; // Input feature dimension (embedDim of previous stage)
  final int outDim; // Output feature dimension (e.g., 2 * inDim)
  final Layer projection;
  final LayerNorm norm;

  PatchMerging({
    required this.inDim,
  })  : outDim = 2 * inDim, // Typically doubles the dimension
        // A 2x2 window means 4 patches are merged, so the input to proj is 4 * inDim
        projection = Layer.fromNeurons(4 * inDim, 2 * inDim),
        norm = LayerNorm(4 * inDim); // Normalize before projection

  /// Forward pass for PatchMerging.
  ///
  /// Takes a list of `ValueVector`s representing patch embeddings,
  /// assuming they form a 2D grid that can be merged.
  /// `x` should effectively be (H_grid * W_grid, inDim)
  ///
  /// This pseudo-code simplifies the 2D grid reshaping for merging.
  /// In reality, you'd need the original H and W of the feature map.
  List<ValueVector> forward(List<ValueVector> x, int H, int W) {
    // H, W are the current height and width of the feature map in terms of patches.
    // e.g., if image is 224x224 and patch size is 4, initial H=56, W=56.
    // After first stage, H=28, W=28.

    assert(H % 2 == 0 && W % 2 == 0,
        "H and W must be divisible by 2 for 2x2 merging.");
    assert(x.length == H * W, "Input length must match H * W.");

    final mergedPatches = <ValueVector>[];
    for (int i = 0; i < H ~/ 2; i++) {
      for (int j = 0; j < W ~/ 2; j++) {
        // Collect 2x2 neighboring patches
        final p0 = x[i * 2 * W + j * 2];
        final p1 = x[i * 2 * W + (j * 2 + 1)];
        final p2 = x[(i * 2 + 1) * W + j * 2];
        final p3 = x[(i * 2 + 1) * W + (j * 2 + 1)];

        // Concatenate their features
        final concatenated = ValueVector(
            [...p0.values, ...p1.values, ...p2.values, ...p3.values]);

        // Normalize the concatenated features
        final normalized = norm.forward(concatenated);

        // Project to the new dimension
        final projected = projection.forward(normalized);
        mergedPatches.add(projected);
      }
    }
    return mergedPatches; // New size: (H/2 * W/2, outDim)
  }

  @override
  List<Value> parameters() {
    return [...projection.parameters(), ...norm.parameters()];
  }
}
