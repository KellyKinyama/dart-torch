// In a new file: swin_stage.dart

import '/nn/module.dart';
import 'swin_transformer_block.dart';
import 'patch_merging.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A single stage of the Swin Transformer.
/// Consists of multiple SwinTransformerBlocks and optional PatchMerging.
class SwinStage extends Module {
  final List<SwinTransformerBlock> blocks;
  final PatchMerging? patchMerging; // Optional, not for the last stage
  final int windowSize;
  final int embedDim;
  final int numHeads;

  SwinStage({
    required this.embedDim,
    required int depth, // Number of blocks in this stage
    required this.numHeads,
    required this.windowSize,
    bool doPatchMerging = true,
    int? inDimForMerging, // The input dimension to PatchMerging
  })  : blocks = List.generate(depth, (i) {
          // Alternate between W-MSA and SW-MSA
          final isShiftedWindow = i % 2 == 1;
          return SwinTransformerBlock(
            embedSize: embedDim,
            numHeads: numHeads,
            isShiftedWindow: isShiftedWindow,
          );
        }),
        patchMerging = doPatchMerging
            ? PatchMerging(inDim: inDimForMerging ?? embedDim)
            : null;

  /// Forward pass for a Swin Transformer Stage.
  ///
  /// `x` is the input feature map (flattened list of ValueVectors).
  /// `H`, `W` are the current spatial dimensions of the feature map (in patches).
  List<ValueVector> forward(List<ValueVector> x, int H, int W) {
    // Process through all blocks in this stage
    for (int i = 0; i < blocks.length; i++) {
      final isShifted = i % 2 == 1; // Alternate W-MSA and SW-MSA
      x = blocks[i].forward(x, H, W, windowSize, isShifted);
    }

    // Apply patch merging if it exists for this stage
    if (patchMerging != null) {
      x = patchMerging!.forward(x, H, W);
      H = H ~/ 2; // Update H for the next stage
      W = W ~/ 2; // Update W for the next stage
    }

    return x; // Return transformed features
  }

  @override
  List<Value> parameters() {
    return [
      ...blocks.expand((b) => b.parameters()),
      if (patchMerging != null) ...patchMerging!.parameters(),
    ];
  }
}