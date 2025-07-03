// file: example_swin_transformer.dart

import 'dart:math' as math; // For general math operations if needed

// Import your custom Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/module.dart';
import '/nn/layer.dart'; // Assuming you have a basic Layer (linear)

// Import Swin Transformer components
import 'patch_embedding.dart';
import 'patch_merging.dart';
import 'window_attention.dart';
import '../transformer2/feed_forward.dart'; // Re-used from your existing FFN
import '../transformer2/layer_norm2.dart'; // Re-used from your existing LayerNorm
import 'swin_transformer_block.dart';
import 'swin_stage.dart';
import 'swin_transformer.dart';

void main() {
  print("--- Swin Transformer Example Usage ---");

  // 1. Define Model Hyperparameters (Example values, typically from Swin-T, S, B, L configs)
  const int imageSize = 224; // e.g., 224x224 input image
  const int patchSize = 4;   // e.g., 4x4 initial patches
  const int inChannels = 3;  // RGB image
  const int windowSize = 7;  // Window size for attention (e.g., 7x7)

  // Swin-T (Tiny) configuration parameters for example
  // These lists define the embedding dimensions, number of blocks (depth), and heads for each stage.
  const List<int> embedDims = [96, 192, 384, 768]; // C, 2C, 4C, 8C
  const List<int> depths = [2, 2, 6, 2];          // Number of Swin Transformer blocks in each of 4 stages
  const List<int> numHeads = [3, 6, 12, 24];      // Number of attention heads in each stage
  const int numClasses = 1000;                    // Example: For ImageNet classification

  // Calculate the initial number of patches (height and width)
  final int initialNumPatchesH = imageSize ~/ patchSize;
  final int initialNumPatchesW = imageSize ~/ patchSize;
  final int totalInitialPatches = initialNumPatchesH * initialNumPatchesW;

  print("Model Configuration:");
  print("  Image Size: $imageSize x $imageSize");
  print("  Patch Size: $patchSize x $patchSize");
  print("  Initial Patch Grid: $initialNumPatchesH x $initialNumPatchesW ($totalInitialPatches patches)");
  print("  Window Size: $windowSize x $windowSize");
  print("  Embed Dims (Stages): $embedDims");
  print("  Depths (Blocks per Stage): $depths");
  print("  Num Heads (per Stage): $numHeads");
  print("  Number of Classes (for classification head): $numClasses");

  // 2. Instantiate the SwinTransformer model
  print("\nInitializing SwinTransformer...");
  final model = SwinTransformer(
    imageSize: imageSize,
    patchSize: patchSize,
    inChannels: inChannels,
    embedDims: embedDims,
    depths: depths,
    numHeads: numHeads,
    windowSize: windowSize,
    numClasses: numClasses,
  );
  print("SwinTransformer initialized. Total parameters: ${model.parameters().length}");

  // 3. Prepare a dummy image input
  // Simulate a flattened image, where each "pixel" is a Value (float).
  // For a 224x224 RGB image, total pixels = 224*224*3
  // And each patch is 4x4x3 = 48 values.
  // The PatchEmbedding expects a List<ValueVector>, where each ValueVector
  // represents a flattened patch.
  print("\nPreparing dummy image input...");
  final List<ValueVector> dummyImagePatches = List.generate(
    totalInitialPatches, // Number of patches
    (patchIdx) => ValueVector(
      List.generate(
        patchSize * patchSize * inChannels, // Size of a flattened patch
        (valIdx) => Value(math.Random().nextDouble() * 0.1), // Small random values
      ),
    ),
  );
  print("Dummy image input created: ${dummyImagePatches.length} patches, each with ${dummyImagePatches[0].values.length} features.");

  // 4. Perform a forward pass
  print("\nPerforming forward pass...");
  try {
    final List<ValueVector> outputFeatures = model.forward(
      dummyImagePatches,
      imageSize,
      imageSize,
    );
    print("Forward pass successful!");

    // The output features `outputFeatures` will be the feature map from the last stage.
    // Its dimensions will be: (H_final_patches * W_final_patches, embedDims.last)
    // For Swin-T with 224x224, patch=4, after 4 stages of 2x downsampling:
    // H_final_patches = 224 / 4 / (2^3) = 56 / 8 = 7
    // W_final_patches = 224 / 4 / (2^3) = 56 / 8 = 7
    // So, it should be 7 * 7 = 49 patches, each with embedDims.last (768) features.
    print("Output Features:");
    print("  Number of output tokens (patches): ${outputFeatures.length}");
    if (outputFeatures.isNotEmpty) {
      print("  Feature dimension per token: ${outputFeatures[0].values.length}");
      print("  Expected final patch grid size: ${imageSize ~/ (patchSize * math.pow(2, depths.length -1))} x ${imageSize ~/ (patchSize * math.pow(2, depths.length -1))} = ${outputFeatures.length}");
      print("  Expected feature dimension: ${embedDims.last}");
    }

    // You would then typically add a classification head (linear layer + softmax)
    // for classification tasks or other heads for detection/segmentation.
    // For instance, a global average pooling could be applied to `outputFeatures`
    // followed by a linear layer to `numClasses`.

  } catch (e) {
    print("An error occurred during forward pass: $e");
    print("Please ensure all dependencies (Value, ValueVector, Module, Layer, etc.) are correctly implemented and imported.");
  }

  print("\n--- End of Swin Transformer Example ---");
}