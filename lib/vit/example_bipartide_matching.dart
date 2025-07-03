// file: example_object_detection.dart

import 'dart:math';

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';

// Import your new object detection components
import 'object_detector2.dart'; // Your combined detector

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
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
      3; // Fixed number of object predictions the model will output

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
  // Each map represents one ground truth object: {'bbox': [x,y,w,h], 'class_id': int}
  final List<Map<String, dynamic>> gtObjects = [
    {
      'bbox': [0.1, 0.1, 0.2, 0.2],
      'class_id': random.nextInt(numClasses)
    },
    {
      'bbox': [0.5, 0.5, 0.3, 0.3],
      'class_id': random.nextInt(numClasses)
    },
    // Add more if you want to test with more GT objects, up to numQueries
  ];

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");
  print(
      "Ground Truth Objects: ${gtObjects.map((obj) => 'Bbox: ${obj['bbox'].map((v) => v.toStringAsFixed(2)).toList()}, Class: ${obj['class_id']}').toList()}");

  // --- Helper for calculating cost between a predicted object and a ground truth object ---
  // This cost is used for bipartite matching.
  Value calculatePairwiseCost(ValueVector predBbox, ValueVector predLogits,
      List<double> gtBbox, int gtClassId, int numClasses) {
    // Bounding Box Cost (L1 Loss)
    Value bboxCost = Value(0.0);
    for (int i = 0; i < 4; i++) {
      bboxCost += (predBbox.values[i] - Value(gtBbox[i])).abs();
    }
    bboxCost = bboxCost / Value(4.0); // Average L1 cost

    // Classification Cost (Negative Log-Likelihood of the true class)
    // For classification, we want to maximize the probability of the true class.
    // So, we minimize the negative log-probability.
    // First, convert logits to log-probabilities (log_softmax)
    final List<Value> logProbs =
        predLogits.softmax().values.map((v) => v.log()).toList();
    // Cost is negative log-prob of the true class
    final Value classCost = -logProbs[gtClassId];

    // Total cost (weighted sum, these weights are hyper-parameters)
    // You might use different weights for bbox and class costs.
    final Value totalPairCost = bboxCost * Value(1.0) + classCost * Value(1.0);
    return totalPairCost;
  }

  // --- Training Loop with Simplified Greedy Bipartite Matching ---
  final epochs = 200; // Increased epochs for more complex task
  print("\nTraining Multi-Object Detector for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, List<ValueVector>> predictions =
        detector.forward(dummyImageData);
    final List<ValueVector> predictedBboxes = predictions['boxes']!;
    final List<ValueVector> predictedLogits = predictions['logits']!;

    // 2. Bipartite Matching (Simplified Greedy Approach)
    // This finds the best assignment between predicted and ground truth objects.
    // A real Hungarian algorithm would be used here.

    // Keep track of which predicted queries and GT objects have been matched
    final Set<int> matchedPredIndices = {};
    final Set<int> matchedGtIndices = {};
    final Map<int, int> assignments = {}; // {predicted_idx: gt_idx}

    // Calculate cost matrix for all possible (pred, gt) pairs
    final List<List<Value>> costMatrix = List.generate(numQueries,
        (_) => List.generate(gtObjects.length, (_) => Value(double.infinity)));

    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      for (int gIdx = 0; gIdx < gtObjects.length; gIdx++) {
        costMatrix[pIdx][gIdx] = calculatePairwiseCost(
          predictedBboxes[pIdx],
          predictedLogits[pIdx],
          gtObjects[gIdx]['bbox'] as List<double>,
          gtObjects[gIdx]['class_id'] as int,
          numClasses,
        );
      }
    }

    // Greedy matching: find the lowest cost pair, assign, and remove from consideration
    // This is a simple approximation of Hungarian algorithm.
    while (matchedPredIndices.length < numQueries &&
        matchedGtIndices.length < gtObjects.length) {
      Value minCost = Value(double.infinity);
      int bestPredIdx = -1;
      int bestGtIdx = -1;

      for (int pIdx = 0; pIdx < numQueries; pIdx++) {
        if (matchedPredIndices.contains(pIdx)) continue; // Already matched

        for (int gIdx = 0; gIdx < gtObjects.length; gIdx++) {
          if (matchedGtIndices.contains(gIdx)) continue; // Already matched

          if (costMatrix[pIdx][gIdx].data < minCost.data) {
            minCost = costMatrix[pIdx][gIdx];
            bestPredIdx = pIdx;
            bestGtIdx = gIdx;
          }
        }
      }

      if (bestPredIdx != -1 && bestGtIdx != -1) {
        assignments[bestPredIdx] = bestGtIdx;
        matchedPredIndices.add(bestPredIdx);
        matchedGtIndices.add(bestGtIdx);
      } else {
        // No more matches possible
        break;
      }
    }

    // 3. Calculate Loss based on Assignments
    Value totalLoss = Value(0.0);

    // Loss for matched objects
    for (var entry in assignments.entries) {
      final int predIdx = entry.key;
      final int gtIdx = entry.value;

      final ValueVector currentPredictedBbox = predictedBboxes[predIdx];
      final ValueVector currentPredictedLogits = predictedLogits[predIdx];
      final List<double> currentGtBboxCoords =
          gtObjects[gtIdx]['bbox'] as List<double>;
      final int currentGtClassId = gtObjects[gtIdx]['class_id'] as int;

      // Bounding Box Loss (L1 Loss)
      Value bboxLoss = Value(0.0);
      for (int i = 0; i < 4; i++) {
        bboxLoss +=
            (currentPredictedBbox.values[i] - Value(currentGtBboxCoords[i]))
                .abs();
      }
      bboxLoss = bboxLoss / Value(4.0);

      // Classification Loss (Cross-Entropy for matched class)
      final gtClassVector = ValueVector(List.generate(
        numClasses + 1,
        (i) => Value(i == currentGtClassId ? 1.0 : 0.0),
      ));
      final classLoss =
          currentPredictedLogits.softmax().crossEntropy(gtClassVector);

      totalLoss += bboxLoss + classLoss;
    }

    // Loss for unmatched predicted objects (they should predict background)
    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      if (!matchedPredIndices.contains(pIdx)) {
        final ValueVector currentPredictedLogits = predictedLogits[pIdx];
        // Target is background class
        final gtBackgroundClassVector = ValueVector(List.generate(
          numClasses + 1,
          (i) => Value(
              i == numClasses ? 1.0 : 0.0), // numClasses is the background ID
        ));
        final backgroundClassLoss = currentPredictedLogits
            .softmax()
            .crossEntropy(gtBackgroundClassVector);
        totalLoss += backgroundClassLoss;
        // No bounding box loss for background predictions
      }
    }

    // 4. Backward pass and optimization step
    detector.zeroGrad(); // Clear gradients
    totalLoss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 20 == 0 || epoch == epochs - 1) {
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
      "\nNote: This example demonstrates multi-object output and a simplified matching. For real-world accuracy, "
      "you'd need a robust bipartite matching algorithm (e.g., Hungarian algorithm) during training, "
      "and potentially Non-Maximum Suppression (NMS) during inference if the model doesn't inherently "
      "avoid duplicate predictions (like DETR does with its matching).");
}
