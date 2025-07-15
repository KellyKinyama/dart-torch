// file: example_bipartide_matching.dart

import 'dart:math';

// Import your core Value and Module system
import '../algorithms/hungarian_algorithm.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

// Import your new object detection components
import 'object_detector2.dart'; // Your combined detector
// Import the Hungarian Algorithm

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
  print(
      "--- ViT-based Multi-Object Detection Example with Hungarian Matching ---"); //

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

  print("Detector Configuration:"); //
  print("  Image Size: $imageSize x $imageSize"); //
  print("  Patch Size: $patchSize x $patchSize"); //
  print("  Embed Size: $embedSize"); //
  print("  Num Classes: $numClasses"); //
  print("  Num Queries (Max Objects Predicted): $numQueries"); //

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

  final optimizer = SGD(detector.parameters(), 0.01); //

  // --- Dummy Image Data and Ground Truth ---
  // For a single image, we'll simulate multiple ground truth objects.
  final int totalPixels = imageSize * imageSize * numChannels; //
  final Random random = Random(); //

  // Dummy image data
  final List<double> dummyImageData =
      List.generate(totalPixels, (i) => random.nextDouble()); //

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
    // If gtObjects.length > numQueries, some GT objects will be unmatched.
    // If gtObjects.length < numQueries, some predicted queries will be unmatched (assigned to background).
  ];

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}..."); //
  print(
      "Ground Truth Objects: ${gtObjects.map((obj) => 'Bbox: ${obj['bbox'].map((v) => v.toStringAsFixed(2)).toList()}, Class: ${obj['class_id']}').toList()}"); //

  // --- Helper for calculating cost between a predicted object and a ground truth object ---
  // This cost is used for bipartite matching.
  // We explicitly make this return an `int` for the Hungarian algorithm.
  // Floating point costs need to be scaled and converted to integers.
  int calculatePairwiseCost(ValueVector predBbox, ValueVector predLogits,
      List<double> gtBbox, int gtClassId, int numClasses) {
    // Bounding Box Cost (L1 Loss)
    Value bboxCost = Value(0.0); //
    for (int i = 0; i < 4; i++) {
      //
      bboxCost += (predBbox.values[i] - Value(gtBbox[i])).abs(); //
    }
    bboxCost = bboxCost / Value(4.0); // Average L1 cost

    // Classification Cost (Negative Log-Likelihood of the true class)
    // For classification, we want to maximize the probability of the true class.
    // So, we minimize the negative log-probability.
    // First, convert logits to log-probabilities (log_softmax)
    final List<Value> logProbs =
        predLogits.softmax().values.map((v) => v.log()).toList(); //
    // Cost is negative log-prob of the true class
    final Value classCost = -logProbs[gtClassId]; //

    // Total cost (weighted sum, these weights are hyper-parameters)
    // You might use different weights for bbox and class costs.
    // Scale the floating point costs to integers for the Hungarian algorithm.
    // A scaling factor of 10000 or 100000 is common to maintain precision.
    final double scalingFactor = 100000.0;
    final Value totalPairCostValue =
        bboxCost * Value(1.0) + classCost * Value(1.0); //
    return (totalPairCostValue.data * scalingFactor).round();
  }

  // --- Training Loop with Hungarian Bipartite Matching ---
  final epochs = 200; // Increased epochs for more complex task
  print("\nTraining Multi-Object Detector for $epochs epochs..."); //

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, List<ValueVector>> predictions =
        detector.forward(dummyImageData); //
    final List<ValueVector> predictedBboxes = predictions['boxes']!; //
    final List<ValueVector> predictedLogits = predictions['logits']!; //

    // 2. Bipartite Matching using Hungarian Algorithm
    // Create a cost matrix for the Hungarian algorithm.
    // The matrix dimensions will be max(numQueries, num_gt_objects) x max(numQueries, num_gt_objects).
    // We need to pad with dummy entries to make it square if queries != GT objects.
    // A large cost is used for padding unmatched pairs.
    final int effectiveDim = max(numQueries, gtObjects.length);
    final int largeCost = 999999999; // A very large cost for unmatched pairs

    final List<List<int>> hungarianCostMatrix = List.generate(
        effectiveDim,
        (_) => List.generate(
            effectiveDim, (_) => largeCost)); // Initialize with high costs

    // Populate the cost matrix for actual (pred, gt) pairs
    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      for (int gIdx = 0; gIdx < gtObjects.length; gIdx++) {
        hungarianCostMatrix[pIdx][gIdx] = calculatePairwiseCost(
          predictedBboxes[pIdx],
          predictedLogits[pIdx],
          gtObjects[gIdx]['bbox'] as List<double>,
          gtObjects[gIdx]['class_id'] as int,
          numClasses,
        );
      }
    }

    // Instantiate and run the Hungarian Algorithm
    final hungarian = HungarianAlgorithm(hungarianCostMatrix); //
    final List<int> assignments =
        hungarian.getAssignment(); // pred_idx -> gt_idx (or dummy)

    // 3. Calculate Loss based on Assignments
    Value totalLoss = Value(0.0); //

    // Track which GT objects were actually matched
    final Set<int> matchedGtIndices = {}; //

    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      final int assignedGtIdx = assignments[pIdx];

      // If `assignedGtIdx` corresponds to an actual GT object (within gtObjects.length)
      // AND that GT object hasn't been matched by another (potentially lower cost) query
      // (The Hungarian algorithm *should* handle this uniqueness, but a double check is good for understanding)
      if (assignedGtIdx < gtObjects.length) {
        // This is a matched pair (predicted_query -> actual_ground_truth_object)
        matchedGtIndices.add(assignedGtIdx); // Mark this GT as matched

        final ValueVector currentPredictedBbox = predictedBboxes[pIdx]; //
        final ValueVector currentPredictedLogits = predictedLogits[pIdx]; //
        final List<double> currentGtBboxCoords =
            gtObjects[assignedGtIdx]['bbox'] as List<double>; //
        final int currentGtClassId =
            gtObjects[assignedGtIdx]['class_id'] as int; //

        // Bounding Box Loss (L1 Loss)
        Value bboxLoss = Value(0.0); //
        for (int i = 0; i < 4; i++) {
          //
          bboxLoss +=
              (currentPredictedBbox.values[i] - Value(currentGtBboxCoords[i]))
                  .abs(); //
        }
        bboxLoss = bboxLoss / Value(4.0); //

        // Classification Loss (Cross-Entropy for matched class)
        // Here, the background class is `numClasses`. If `currentGtClassId` is one of the
        // actual object classes (0 to numClasses-1), then it's a true object.
        final gtClassVector = ValueVector(List.generate(
          numClasses + 1, // +1 for background class
          (i) => Value(i == currentGtClassId ? 1.0 : 0.0), //
        ));
        final classLoss =
            currentPredictedLogits.softmax().crossEntropy(gtClassVector); //

        totalLoss += bboxLoss + classLoss; //
      } else {
        // This predicted query was assigned to a "dummy" GT object (beyond actual gtObjects.length).
        // This means it's an unmatched prediction, and it should predict background.
        final ValueVector currentPredictedLogits = predictedLogits[pIdx]; //
        // Target is background class
        final gtBackgroundClassVector = ValueVector(List.generate(
          numClasses + 1,
          (i) => Value(
              i == numClasses ? 1.0 : 0.0), // numClasses is the background ID
        ));
        final backgroundClassLoss = currentPredictedLogits
            .softmax()
            .crossEntropy(gtBackgroundClassVector); //
        totalLoss += backgroundClassLoss; //
        // No bounding box loss for background predictions
      }
    }

    // Handle unmatched ground truth objects (if gtObjects.length > numQueries)
    // For simplicity in this example, we assume that if gtObjects.length > numQueries,
    // the model effectively "misses" some objects and their loss isn't directly computed
    // through the Hungarian assignment. In a full DETR, there would be a "no object"
    // prediction for queries not matched to GT, and a loss associated with that.
    // However, the Hungarian algorithm implicitly handles this by trying to match as many
    // as possible for the minimum cost. If a GT object cannot be matched to any query
    // without incurring a higher cost than matching a query to background, it remains
    // "unmatched" from the queries' perspective. For a simplified training, we just focus
    // on the losses for the predictions.

    // 4. Backward pass and optimization step
    detector.zeroGrad(); // Clear gradients
    totalLoss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 20 == 0 || epoch == epochs - 1) {
      //
      print(
          "Epoch $epoch | Total Loss: ${totalLoss.data.toStringAsFixed(4)}"); //
    }
  }
  print(
      "✅ Multi-Object Detector training complete with Hungarian Matching."); //

  // --- Inference Example ---
  print("\n--- Multi-Object Detector Inference ---"); //
  final List<double> newDummyImageData = List.generate(
      totalPixels, (i) => random.nextDouble()); // A new random image

  print(
      "New Dummy Image Data created (first 10 values): ${newDummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}..."); //

  final Map<String, List<ValueVector>> inferencePredictions =
      detector.forward(newDummyImageData); //
  final List<ValueVector> inferredBboxes = inferencePredictions['boxes']!; //
  final List<ValueVector> inferredLogits = inferencePredictions['logits']!; //

  print("\nInferred Objects:"); //
  for (int q = 0; q < numQueries; q++) {
    //
    final ValueVector currentInferredBbox = inferredBboxes[q]; //
    final ValueVector currentInferredLogits = inferredLogits[q]; //
    final ValueVector currentInferredProbs = currentInferredLogits.softmax(); //

    // Find the predicted class (index with highest probability)
    double maxProb = -1.0; //
    int predictedClass = -1; //
    for (int i = 0; i < currentInferredProbs.values.length; i++) {
      //
      if (currentInferredProbs.values[i].data > maxProb) {
        //
        maxProb = currentInferredProbs.values[i].data; //
        predictedClass = i; //
      }
    }

    print("  Object ${q + 1}:"); //
    print(
        "    Bbox: ${currentInferredBbox.values.map((v) => v.data.toStringAsFixed(4)).toList()}"); //
    print(
        "    Class: $predictedClass (Prob: ${maxProb.toStringAsFixed(4)})"); //
  }

  print(
      "\nNote: This example demonstrates multi-object output with Hungarian matching during training. For real-world accuracy, "
      "you'd also consider a more sophisticated loss weighting, and potentially "
      "Non-Maximum Suppression (NMS) during inference for models that don't inherently "
      "avoid duplicate predictions (like DETR does with its matching)."); //
}
