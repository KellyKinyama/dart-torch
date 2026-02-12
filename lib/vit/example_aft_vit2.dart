// file: example_vit.dart

import 'dart:math';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'aft_vision_transformer.dart'; // Import your ViT

// Assuming SGD is in a shared utility or copied into this file for brevity
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- Vision Transformer Example ---");

  // Model parameters
  final imageSize = 32; // Example: Small 32x32 image
  final patchSize = 8; // Patches will be 8x8 pixels
  final numChannels = 3; // RGB image
  final embedSize = 64; // Transformer embedding dimension
  final numClasses = 10; // E.g., for CIFAR-10 dataset
  final numLayers = 2; // Small number of layers for quick execution
  final numHeads = 4; // Number of attention heads

  // Instantiate the ViT model
  final vit = VisionTransformer(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numClasses: numClasses,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  final optimizer = SGD(vit.parameters(), 0.01);

  // --- Dummy Image Data ---
  // A flattened list of pixel values (0.0 to 1.0)
  // Size: imageSize * imageSize * numChannels
  final int totalPixels = imageSize * imageSize * numChannels;
  final Random random = Random();

  // Create a dummy image data and a target class for training
  final List<double> dummyImageData =
      List.generate(totalPixels, (i) => random.nextDouble());
  final int dummyTargetClass =
      random.nextInt(numClasses); // A random target class

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");
  print("Target Class: $dummyTargetClass");

  // --- Training Loop (Simplified) ---
  final epochs = 50; // Run for a few epochs
  print("\nTraining ViT for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final logits =
        vit.forward(dummyImageData); // Returns a List<Value> for classes

    // 2. Calculate Cross-Entropy Loss
    // Convert target class to one-hot vector (using Value objects)
    final targetVector = ValueVector(List.generate(
      numClasses,
      (i) => Value(i == dummyTargetClass ? 1.0 : 0.0),
    ));

    // Convert logits (List<Value>) to ValueVector for softmax and crossEntropy
    final logitsVector = ValueVector(logits);

    final loss = logitsVector.softmax().crossEntropy(targetVector);

    // 3. Backward pass and optimization step
    vit.zeroGrad(); // Clear gradients
    loss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${loss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ ViT training complete.");

  // --- Inference Example ---
  print("\n--- ViT Inference ---");
  final List<double> newDummyImageData = List.generate(
      totalPixels, (i) => random.nextDouble()); // A new random image

  print(
      "New Dummy Image Data created (first 10 values): ${newDummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");

  final inferenceLogits = vit.forward(newDummyImageData);
  final predictedProbs = ValueVector(inferenceLogits).softmax();

  // Find the predicted class (index with highest probability)
  double maxProb = -1.0;
  int predictedClass = -1;
  for (int i = 0; i < predictedProbs.values.length; i++) {
    if (predictedProbs.values[i].data > maxProb) {
      maxProb = predictedProbs.values[i].data;
      predictedClass = i;
    }
  }

  print(
      "Inference Logits: ${inferenceLogits.map((v) => v.data.toStringAsFixed(4)).toList()}");
  print(
      "Predicted Probabilities: ${predictedProbs.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
  print(
      "Predicted Class: $predictedClass (with probability ${maxProb.toStringAsFixed(4)})");

  print(
      "\nNote: For real-world usage, the `_createPatchesAndEmbeddings` function within ViT would need robust image processing to handle actual image files and their pixel layouts (e.g., from `image` package or custom parsing).");
}
