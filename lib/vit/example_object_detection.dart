// file: example_object_detection.dart

import 'dart:math';

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/module.dart';
import '/nn/layer.dart';

// Import your new object detection components
import 'vit_backbone.dart'; // Your modified ViT backbone
import 'object_detection_head.dart'; // Your new detection head
import 'object_detector.dart'; // Your combined detector

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad!;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

void main() {
  print("--- ViT-based Object Detection Example ---");

  // Model parameters
  final imageSize = 32; // Example: Small 32x32 image
  final patchSize = 8; // Patches will be 8x8 pixels
  final numChannels = 3; // RGB image
  final embedSize = 64; // Transformer embedding dimension
  final numClasses =
      5; // Example: 5 object classes (e.g., car, person, dog, cat, bike)
  final numLayers = 2; // Small number of layers for quick execution
  final numHeads = 4; // Number of attention heads

  // Instantiate the ViTObjectDetector model
  final detector = ViTObjectDetector(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numLayers: numLayers,
    numHeads: numHeads,
    numClasses: numClasses,
  );

  final optimizer = SGD(detector.parameters(), 0.01);

  // --- Dummy Image Data and Ground Truth ---
  // For a single image, we'll simulate one ground truth object.
  // In real detection, you'd have lists of boxes and classes per image.
  final int totalPixels = imageSize * imageSize * numChannels;
  final Random random = Random();

  // Dummy image data
  final List<double> dummyImageData =
      List.generate(totalPixels, (i) => random.nextDouble());

  // Dummy Ground Truth for ONE object:
  // Bounding box: [x_center, y_center, width, height] (normalized 0-1)
  final List<double> gtBboxCoords = [
    random.nextDouble(), // x_center
    random.nextDouble(), // y_center
    random.nextDouble() * 0.5 + 0.1, // width (0.1 to 0.6)
    random.nextDouble() * 0.5 + 0.1, // height (0.1 to 0.6)
  ];
  // Class label (0 to numClasses-1, or numClasses for background)
  final int gtClassId = random.nextInt(numClasses); // A random object class

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");
  print(
      "Ground Truth Bbox: ${gtBboxCoords.map((v) => v.toStringAsFixed(2)).toList()}");
  print("Ground Truth Class: $gtClassId");

  // --- Training Loop (Highly Simplified for ONE object) ---
  final epochs = 100; // Run for a few epochs
  print("\nTraining Object Detector for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, ValueVector> predictions =
        detector.forward(dummyImageData);
    final ValueVector predictedBbox = predictions['boxes']!;
    final ValueVector predictedLogits = predictions['logits']!;

    // 2. Calculate Loss
    // a. Bounding Box Loss (L1 Loss)
    Value bboxLoss = Value(0.0);
    for (int i = 0; i < 4; i++) {
      bboxLoss += (predictedBbox.values[i] - Value(gtBboxCoords[i]));
    }
    bboxLoss = bboxLoss / Value(4.0); // Average L1 loss

    // b. Classification Loss (Cross-Entropy)
    // Convert ground truth class to one-hot vector (including background)
    final gtClassVector = ValueVector(List.generate(
      numClasses + 1, // +1 for background class
      (i) => Value(i == gtClassId ? 1.0 : 0.0),
    ));
    final classLoss = predictedLogits.softmax().crossEntropy(gtClassVector);

    // Total loss (simple sum, in real detectors, weights are used)
    final totalLoss = bboxLoss + classLoss;

    // 3. Backward pass and optimization step
    detector.zeroGrad(); // Clear gradients
    totalLoss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 1 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Total Loss: ${totalLoss.data.toStringAsFixed(4)} "
          "(Bbox Loss: ${bboxLoss.data.toStringAsFixed(4)}, "
          "Class Loss: ${classLoss.data.toStringAsFixed(4)})");
    }
  }
  print("✅ Object Detector training complete.");

  // --- Inference Example ---
  print("\n--- Object Detector Inference ---");
  final List<double> newDummyImageData = List.generate(
      totalPixels, (i) => random.nextDouble()); // A new random image

  print(
      "New Dummy Image Data created (first 10 values): ${newDummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");

  final Map<String, ValueVector> inferencePredictions =
      detector.forward(newDummyImageData);
  final ValueVector inferredBbox = inferencePredictions['boxes']!;
  final ValueVector inferredLogits = inferencePredictions['logits']!;
  final ValueVector inferredProbs = inferredLogits.softmax();

  // Find the predicted class (index with highest probability)
  double maxProb = -1.0;
  int predictedClass = -1;
  for (int i = 0; i < inferredProbs.values.length; i++) {
    if (inferredProbs.values[i].data > maxProb) {
      maxProb = inferredProbs.values[i].data;
      predictedClass = i;
    }
  }

  print(
      "Inferred Bbox: ${inferredBbox.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
  print(
      "Inferred Class Probabilities: ${inferredProbs.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
  print(
      "Predicted Class: $predictedClass (with probability ${maxProb.toStringAsFixed(4)})");

  print(
      "\nNote: This is a highly simplified object detection example. Real-world detectors handle multiple objects, non-maximum suppression, and more complex loss functions and evaluation metrics.");
}
