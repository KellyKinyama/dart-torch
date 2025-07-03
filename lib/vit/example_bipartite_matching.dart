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

// --- CONCEPTUAL HUNGARIAN ALGORITHM IMPLEMENTATION ---
// This is a placeholder. A full implementation of the Hungarian algorithm
// with Value objects would be extremely complex and is typically
// handled by highly optimized numerical libraries (e.g., SciPy's linear_sum_assignment).
// The purpose here is to demonstrate its integration point and expected output.
Map<int, int> _hungarianAlgorithm(List<List<Value>> costMatrix) {
  // costMatrix: [num_queries][num_gt_objects] where each element is a Value representing cost.
  // Returns: {predicted_idx: gt_idx} for optimal matches.

  // In a real scenario, this function would implement the steps of the Hungarian algorithm:
  // 1. Subtract the smallest element from each row.
  // 2. Subtract the smallest element from each column.
  // 3. Cover all zeros with a minimum number of lines.
  // 4. If number of lines < matrix size, uncover and subtract smallest uncovered element.
  // 5. Repeat until all zeros are covered.
  // 6. Find optimal assignment from covered zeros.

  // For demonstration, we'll use a very simplified greedy approach
  // that mimics the *output format* of Hungarian, but is not optimal.
  // This is to allow the rest of the training loop to function.
  // If you were to implement this for real, you'd need a robust
  // algorithm for finding minimum weight perfect matching.

  final int numPredictions = costMatrix.length;
  final int numGroundTruths = costMatrix.isNotEmpty ? costMatrix[0].length : 0;

  final Set<int> matchedPredIndices = {};
  final Set<int> matchedGtIndices = {};
  final Map<int, int> assignments = {}; // {predicted_idx: gt_idx}

  // Create a list of all possible (pred_idx, gt_idx, cost_value) tuples
  final List<Map<String, dynamic>> allCosts = [];
  for (int pIdx = 0; pIdx < numPredictions; pIdx++) {
    for (int gIdx = 0; gIdx < numGroundTruths; gIdx++) {
      allCosts.add(
          {'pred_idx': pIdx, 'gt_idx': gIdx, 'cost': costMatrix[pIdx][gIdx]});
    }
  }

  // Sort costs in ascending order (greedy choice)
  allCosts.sort((a, b) => a['cost'].data.compareTo(b['cost'].data));

  // Iterate through sorted costs to find greedy matches
  for (var costEntry in allCosts) {
    final int pIdx = costEntry['pred_idx'];
    final int gIdx = costEntry['gt_idx'];

    // If both prediction and ground truth are not yet matched, form a match
    if (!matchedPredIndices.contains(pIdx) &&
        !matchedGtIndices.contains(gIdx)) {
      assignments[pIdx] = gIdx;
      matchedPredIndices.add(pIdx);
      matchedGtIndices.add(gIdx);
    }
  }

  // This simplified greedy matching will work for small examples,
  // but it is NOT the mathematically optimal Hungarian algorithm.
  // A true Hungarian algorithm implementation is much more involved.

  return assignments;
}
// --- END CONCEPTUAL HUNGARIAN ALGORITHM IMPLEMENTATION ---

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
      5; // Fixed number of object predictions the model will output (increased for more flexibility)

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
  // We'll simulate 2-3 objects for this example.
  final List<Map<String, dynamic>> gtObjects = [
    {
      'bbox': [0.1, 0.1, 0.2, 0.2],
      'class_id': random.nextInt(numClasses)
    }, // Object 1
    {
      'bbox': [0.5, 0.5, 0.3, 0.3],
      'class_id': random.nextInt(numClasses)
    }, // Object 2
    {
      'bbox': [0.8, 0.2, 0.15, 0.25],
      'class_id': random.nextInt(numClasses)
    }, // Object 3
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
    // Ensure gtClassId is within bounds of logProbs list
    if (gtClassId >= logProbs.length || gtClassId < 0) {
      // This case should ideally not happen with correct data, but for robustness
      // if gtClassId is out of bounds, assign a high cost.
      return Value(double.infinity);
    }
    final Value classCost = -logProbs[gtClassId];

    // Total cost (weighted sum, these weights are hyper-parameters)
    // You might use different weights for bbox and class costs.
    final Value totalPairCost = bboxCost * Value(1.0) + classCost * Value(1.0);
    return totalPairCost;
  }

  // --- Training Loop with Conceptual Hungarian Matching ---
  final epochs = 400; // Increased epochs for more complex task
  print("\nTraining Multi-Object Detector for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, List<ValueVector>> predictions =
        detector.forward(dummyImageData);
    final List<ValueVector> predictedBboxes = predictions['boxes']!;
    final List<ValueVector> predictedLogits = predictions['logits']!;

    // 2. Prepare Cost Matrix for Hungarian Algorithm
    // The cost matrix dimensions are (num_queries x num_gt_objects)
    final List<List<Value>> costMatrix = List.generate(
        numQueries,
        (_) => List.generate(gtObjects.length,
            (_) => Value(0.0))); // Initialize with dummy Value

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

    // 3. Perform Bipartite Matching (Conceptual Hungarian Algorithm)
    // This function returns the optimal assignments: {predicted_idx: gt_idx}
    final Map<int, int> assignments = _hungarianAlgorithm(costMatrix);

    // 4. Calculate Loss based on Assignments
    Value totalLoss = Value(0.0);

    // Loss for matched objects
    final Set<int> matchedPredIndices = assignments.keys.toSet();

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
        // Target is background class (numClasses is the background ID)
        final gtBackgroundClassVector = ValueVector(List.generate(
          numClasses + 1,
          (i) => Value(i == numClasses ? 1.0 : 0.0),
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

    // Only print if the predicted class is not background and probability is high enough
    if (predictedClass != numClasses && maxProb > 0.5) {
      // Threshold for displaying
      print("  Object ${q + 1}:");
      print(
          "    Bbox: ${currentInferredBbox.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
      print("    Class: $predictedClass (Prob: ${maxProb.toStringAsFixed(4)})");
    } else if (predictedClass == numClasses && maxProb > 0.5) {
      print(
          "  Object ${q + 1}: Predicted Background (Prob: ${maxProb.toStringAsFixed(4)})");
    } else {
      print(
          "  Object ${q + 1}: Low confidence prediction (Class: $predictedClass, Prob: ${maxProb.toStringAsFixed(4)}) - Likely background or noise");
    }
  }

  print(
      "\nNote: This example demonstrates multi-object output and a conceptual Hungarian matching. For real-world accuracy, "
      "a mathematically optimal bipartite matching algorithm (e.g., a full Hungarian algorithm implementation) is typically used during training, "
      "and specialized Non-Maximum Suppression (NMS) or similar post-processing might be needed during inference "
      "if the model doesn't inherently avoid duplicate predictions (like DETR does with its matching).");
}
