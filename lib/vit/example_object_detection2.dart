// file: example_object_detection.dart

import 'dart:math';

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
// import '/nn/module.dart';
// import '/nn/layer.dart';

// Import your new object detection components
// import 'vit_backbone.dart'; // Your modified ViT backbone
// import 'object_detection_head.dart'; // Your new detection head
import 'object_detector2.dart'; // Your combined detector

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
  print("--- ViT-based Multi-Object Detection Example ---");

  // Model parameters
  final imageSize = 32; // Example: Small 32x32 image
  final patchSize = 8; // Patches will be 8x8 pixels
  final numChannels = 3; // RGB image
  final embedSize = 64; // Transformer embedding dimension
  final numClasses =
      5; // Example: 5 object classes (e.g., car, person, dog, cat, bike)
  final numLayers = 2; // Small number of layers for quick execution
  final numHeads = 4; // Number of attention heads
  final numQueries =
      3; // NEW: Fixed number of object predictions the model will output

  print("Detector Configuration:");
  print("  Image Size: $imageSize x $imageSize");
  print("  Patch Size: $patchSize x $patchSize");
  print("  Embed Size: $embedSize");
  print("  Num Classes: $numClasses");
  print("  Num Queries (Max Objects Predicted): $numQueries");

  // Instantiate the ViTObjectDetector model
  final detector = ViTObjectDetector(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numLayers: numLayers,
    numHeads: numHeads,
    numClasses: numClasses,
    numQueries: numQueries, // Pass the new parameter
  );

  final optimizer = SGD(detector.parameters(), 0.01);

  // --- Dummy Image Data and Ground Truth ---
  // For a single image, we'll simulate multiple ground truth objects.
  final int totalPixels = imageSize * imageSize * numChannels;
  final Random random = Random();

  // Dummy image data
  final List<double> dummyImageData =
      List.generate(totalPixels, (i) => random.nextDouble());

  // Dummy Ground Truth for MULTIPLE objects:
  // Each inner list is [x_center, y_center, width, height] (normalized 0-1)
  // Each class ID is 0 to numClasses-1.
  // We'll simulate 2 objects for this example, but the model predicts numQueries.
  final List<List<double>> gtBboxCoordsList = [
    [0.1, 0.1, 0.2, 0.2], // Object 1
    [0.5, 0.5, 0.3, 0.3], // Object 2
    // Add more if numQueries is higher, or fewer to simulate background
  ];
  final List<int> gtClassIdList = [
    random.nextInt(numClasses), // Class for object 1
    random.nextInt(numClasses), // Class for object 2
  ];

  // Pad ground truth lists to numQueries with background class and dummy boxes
  // This is a very simple way to handle varying number of objects.
  // In real DETR, this is handled by bipartite matching.
  while (gtBboxCoordsList.length < numQueries) {
    gtBboxCoordsList.add([0.0, 0.0, 0.0, 0.0]); // Dummy box for background
    gtClassIdList.add(numClasses); // Background class ID
  }

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");
  print(
      "Ground Truth Bboxes: ${gtBboxCoordsList.map((bbox) => bbox.map((v) => v.toStringAsFixed(2)).toList()).toList()}");
  print("Ground Truth Classes: $gtClassIdList");

  // --- Training Loop (Highly Simplified Multi-Object Loss) ---
  final epochs = 200; // Increased epochs for more complex task
  print("\nTraining Multi-Object Detector for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, List<ValueVector>> predictions =
        detector.forward(dummyImageData);
    final List<ValueVector> predictedBboxes = predictions['boxes']!;
    final List<ValueVector> predictedLogits = predictions['logits']!;

    // 2. Calculate Loss (Simplified: Sum of losses for all predicted vs. all GT)
    // This is NOT proper bipartite matching. It's a heuristic for demonstration.
    Value totalLoss = Value(0.0);

    for (int q = 0; q < numQueries; q++) {
      final ValueVector currentPredictedBbox = predictedBboxes[q];
      final ValueVector currentPredictedLogits = predictedLogits[q];

      // Find the "best" matching GT object for this predicted query (simplistic)
      // For demonstration, we'll just match predicted query 'q' to GT object 'q'.
      // This only works if numQueries == num_actual_gt_objects.
      // A proper solution requires Hungarian matching.
      final List<double> currentGtBboxCoords = gtBboxCoordsList[q];
      final int currentGtClassId = gtClassIdList[q];

      // a. Bounding Box Loss (L1 Loss)
      Value bboxLoss = Value(0.0);
      for (int i = 0; i < 4; i++) {
        bboxLoss +=
            (currentPredictedBbox.values[i] - Value(currentGtBboxCoords[i]))
                .abs();
      }
      bboxLoss = bboxLoss / Value(4.0); // Average L1 loss

      // b. Classification Loss (Cross-Entropy)
      final gtClassVector = ValueVector(List.generate(
        numClasses + 1, // +1 for background class
        (i) => Value(i == currentGtClassId ? 1.0 : 0.0),
      ));
      final classLoss =
          currentPredictedLogits.softmax().crossEntropy(gtClassVector);

      totalLoss += bboxLoss + classLoss;
    }

    // 3. Backward pass and optimization step
    detector.zeroGrad(); // Clear gradients
    totalLoss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 2 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Total Loss: ${totalLoss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Multi-Object Detector training complete.");

  // --- Inference Example ---
  print("\n--- Multi-Object Detector Inference ---");
  final List<double> newDummyImageData = List.generate(
      totalPixels, (i) => random.nextDouble()); // A new random image

  print(
      "New Dummy Image Data created (first 10 values): ${newDummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");

  final Map<String, List<ValueVector>> inferencePredictions =
      detector.forward(newDummyImageData);
  final List<ValueVector> inferredBboxes = inferencePredictions['boxes']!;
  final List<ValueVector> inferredLogits = inferencePredictions['logits']!;

  print("\nInferred Objects:");
  for (int q = 0; q < numQueries; q++) {
    final ValueVector currentInferredBbox = inferredBboxes[q];
    final ValueVector currentInferredLogits = inferredLogits[q];
    final ValueVector currentInferredProbs = currentInferredLogits.softmax();

    // Find the predicted class (index with highest probability)
    double maxProb = -1.0;
    int predictedClass = -1;
    for (int i = 0; i < currentInferredProbs.values.length; i++) {
      if (currentInferredProbs.values[i].data > maxProb) {
        maxProb = currentInferredProbs.values[i].data;
        predictedClass = i;
      }
    }

    print("  Object ${q + 1}:");
    print(
        "    Bbox: ${currentInferredBbox.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
    print("    Class: $predictedClass (Prob: ${maxProb.toStringAsFixed(4)})");
  }

  print(
      "\nNote: This example demonstrates multi-object output. For real-world accuracy, "
      "you'd need proper bipartite matching during training, and potentially "
      "Non-Maximum Suppression (NMS) during inference if the model doesn't inherently "
      "avoid duplicate predictions (like DETR does with its matching).");
}
