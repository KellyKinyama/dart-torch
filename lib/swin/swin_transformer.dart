// In a new file: swin_transformer.dart

import '/nn/module.dart';
import 'patch_embedding.dart';
import 'swin_stage.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer2/layer_norm2.dart'; // Assuming layer_norm2.dart contains LayerNorm

/// The full Swin Transformer model for computer vision tasks.
class SwinTransformer extends Module {
  final PatchEmbedding patchEmbedding;
  final List<SwinStage> stages;
  final LayerNorm finalNorm; // Optional, often used after last stage
  final int patchSize; // e.g., 4

  SwinTransformer({
    required int imageSize, // e.g., 224
    required this.patchSize, // e.g., 4
    required int inChannels, // e.g., 3
    required List<int> embedDims, // e.g., [96, 192, 384, 768]
    required List<int> depths, // e.g., [2, 2, 6, 2] - num blocks per stage
    required List<int> numHeads, // e.g., [3, 6, 12, 24] - num heads per stage
    required int windowSize, // e.g., 7
    required int numClasses, // For classification head (optional)
  })  : patchEmbedding = PatchEmbedding(
            patchSize: patchSize,
            inChannels: inChannels,
            embedDim: embedDims[0]),
        stages = List.generate(depths.length, (i) {
          final isLastStage = (i == depths.length - 1);
          return SwinStage(
            embedDim: embedDims[i],
            depth: depths[i],
            numHeads: numHeads[i],
            windowSize: windowSize,
            doPatchMerging: !isLastStage, // No merging after the last stage
            inDimForMerging: i > 0
                ? embedDims[i - 1]
                : null, // Pass previous stage's embed dim
          );
        }),
        finalNorm = LayerNorm(embedDims.last); // Normalize output of last stage

  // You might also add a classification head here (e.g., a Layer and softmax)
  // final Layer? classificationHead;

  /// Forward pass for the Swin Transformer.
  ///
  /// `imageFlatPatches` are the flattened patches of the input image.
  /// (num_patches, patchSize*patchSize*inChannels)
  List<ValueVector> forward(
      List<ValueVector> imageFlatPatches, int H_img, int W_img) {
    // Initial patch embedding
    var x =
        patchEmbedding.forward(imageFlatPatches); // (num_patches, embedDims[0])

    // Calculate initial H, W in terms of patches
    int currentH = H_img ~/ patchSize;
    int currentW = W_img ~/ patchSize;

    // Pass through all stages
    for (int i = 0; i < stages.length; i++) {
      x = stages[i].forward(x, currentH, currentW);
      // Update H, W if patch merging occurred in the stage
      if (i < stages.length - 1) {
        // Only if not the last stage (which has merging)
        currentH ~/= 2;
        currentW ~/= 2;
      }
    }

    // Apply final layer normalization
    x = x.map((v) => finalNorm.forward(v)).toList();

    // If for classification, often a global average pooling and a linear head
    // For general backbone, return the feature map.
    return x;
  }

  @override
  List<Value> parameters() {
    return [
      ...patchEmbedding.parameters(),
      ...stages.expand((s) => s.parameters()),
      ...finalNorm.parameters(),
      // if (classificationHead != null) ...classificationHead!.parameters(),
    ];
  }
}
