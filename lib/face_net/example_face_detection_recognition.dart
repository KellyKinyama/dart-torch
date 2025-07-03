// file: example_face_detection_recognition.dart

import 'dart:math';

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/module.dart';
import '/nn/layer.dart';

// Import your new object detection components
import '../vit/vit_backbone.dart';
import 'object_detection_head.dart';
import 'object_detector.dart'; // Now effectively your Face Detection/Recognition model

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
  final int numPredictions = costMatrix.length;
  final int numGroundTruths = costMatrix.isNotEmpty ? costMatrix[0].length : 0;

  final Set<int> matchedPredIndices = {};
  final Set<int> matchedGtIndices = {};
  final Map<int, int> assignments = {}; // {predicted_idx: gt_idx}

  final List<Map<String, dynamic>> allCosts = [];
  for (int pIdx = 0; pIdx < numPredictions; pIdx++) {
    for (int gIdx = 0; gIdx < numGroundTruths; gIdx++) {
      allCosts.add(
          {'pred_idx': pIdx, 'gt_idx': gIdx, 'cost': costMatrix[pIdx][gIdx]});
    }
  }

  allCosts.sort((a, b) => a['cost'].data.compareTo(b['cost'].data));

  for (var costEntry in allCosts) {
    final int pIdx = costEntry['pred_idx'];
    final int gIdx = costEntry['gt_idx'];

    if (!matchedPredIndices.contains(pIdx) &&
        !matchedGtIndices.contains(gIdx)) {
      assignments[pIdx] = gIdx;
      matchedPredIndices.add(pIdx);
      matchedGtIndices.add(gIdx);
    }
  }
  return assignments;
}
// --- END CONCEPTUAL HUNGARIAN ALGORITHM IMPLEMENTATION ---

void main() {
  print("--- ViT-based Face Detection and Recognition Example ---");

  // Model parameters
  final imageSize = 32; // Example: Small 32x32 image
  final patchSize = 8; // Patches will be 8x8 pixels
  final numChannels = 3; // RGB image
  final embedSize = 64; // Transformer embedding dimension
  final numIdentities =
      5; // Number of distinct people/identities to recognize (classes 0 to 4)
  final numLayers = 2;
  final numHeads = 4;
  final numQueries =
      5; // Fixed number of object predictions the model will output
  final embeddingDim = 128; // Dimension of the face embedding for recognition

  print("Model Configuration:");
  print("  Image Size: $imageSize x $imageSize");
  print("  Patch Size: $patchSize x $patchSize");
  print("  Embed Size: $embedSize");
  print("  Num Identities (Classes): $numIdentities");
  print("  Num Queries (Max Objects Predicted): $numQueries");
  print("  Embedding Dimension: $embeddingDim");

  // Instantiate the ViTObjectDetector model (now handling face detection + recognition)
  final faceDetectorRecognizer = ViTObjectDetector(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numLayers: numLayers,
    numHeads: numHeads,
    numClasses: numIdentities, // Pass numIdentities as numClasses
    numQueries: numQueries,
    embeddingDim: embeddingDim, // Pass new parameter
  );

  final optimizer = SGD(faceDetectorRecognizer.parameters(), 0.01);

  // --- Dummy Image Data and Ground Truth ---
  final int totalPixels = imageSize * imageSize * numChannels;
  final Random random = Random();

  // Dummy image data
  final List<double> dummyImageData =
      List.generate(totalPixels, (i) => random.nextDouble());

  // Dummy Ground Truth for MULTIPLE faces:
  // Each map represents one ground truth face:
  // {'bbox': [x,y,w,h], 'class_id': int (identity), 'embedding': List<double>}
  final List<Map<String, dynamic>> gtObjects = [
    {
      'bbox': [0.1, 0.1, 0.2, 0.2],
      'class_id': random.nextInt(numIdentities),
      'embedding': List.generate(
          embeddingDim, (i) => random.nextDouble() * 2 - 1) // Random embedding
    },
    {
      'bbox': [0.5, 0.5, 0.3, 0.3],
      'class_id': random.nextInt(numIdentities),
      'embedding':
          List.generate(embeddingDim, (i) => random.nextDouble() * 2 - 1)
    },
    {
      'bbox': [0.8, 0.2, 0.15, 0.25],
      'class_id': random.nextInt(numIdentities),
      'embedding':
          List.generate(embeddingDim, (i) => random.nextDouble() * 2 - 1)
    },
  ];

  print(
      "Dummy Image Data created (first 10 values): ${dummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");
  print(
      "Ground Truth Objects: ${gtObjects.map((obj) => 'Bbox: ${obj['bbox'].map((v) => v.toStringAsFixed(2)).toList()}, Class: ${obj['class_id']}, Embedding (first 3): ${obj['embedding'].sublist(0, 3).map((v) => v.toStringAsFixed(2)).toList()}...').toList()}");

  // --- Helper for calculating cost between a predicted object and a ground truth object ---
  // This cost is used for bipartite matching.
  Value calculatePairwiseCost(
      ValueVector predBbox,
      ValueVector predLogits,
      ValueVector predEmbedding,
      List<double> gtBbox,
      int gtClassId,
      List<double> gtEmbedding,
      int numIdentities,
      int embeddingDim) {
    // Bounding Box Cost (L1 Loss)
    Value bboxCost = Value(0.0);
    for (int i = 0; i < 4; i++) {
      bboxCost += (predBbox.values[i] - Value(gtBbox[i])).abs();
    }
    bboxCost = bboxCost / Value(4.0); // Average L1 cost

    // Classification Cost (Negative Log-Likelihood of the true class)
    final List<Value> logProbs =
        predLogits.softmax().values.map((v) => v.log()).toList();
    if (gtClassId >= logProbs.length || gtClassId < 0) {
      return Value(double.infinity); // Invalid class ID, assign high cost
    }
    final Value classCost = -logProbs[gtClassId];

    // NEW: Embedding Cost (L1 Loss between embeddings)
    Value embeddingCost = Value(0.0);
    for (int i = 0; i < embeddingDim; i++) {
      embeddingCost += (predEmbedding.values[i] - Value(gtEmbedding[i])).abs();
    }
    embeddingCost = embeddingCost / Value(embeddingDim.toDouble());

    // Total cost (weighted sum)
    // Adjust weights as needed for training balance
    final Value totalPairCost = bboxCost * Value(1.0) +
        classCost * Value(1.0) +
        embeddingCost * Value(0.5); // Added embedding cost
    return totalPairCost;
  }

  // --- Training Loop with Conceptual Hungarian Matching ---
  final epochs = 400; // Increased epochs for more complex task
  print("\nTraining Face Detector and Recognizer for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass
    final Map<String, List<ValueVector>> predictions =
        faceDetectorRecognizer.forward(dummyImageData);
    final List<ValueVector> predictedBboxes = predictions['boxes']!;
    final List<ValueVector> predictedLogits = predictions['logits']!;
    final List<ValueVector> predictedEmbeddings =
        predictions['embeddings']!; // NEW

    // 2. Prepare Cost Matrix for Hungarian Algorithm
    final List<List<Value>> costMatrix = List.generate(
        numQueries, (_) => List.generate(gtObjects.length, (_) => Value(0.0)));

    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      for (int gIdx = 0; gIdx < gtObjects.length; gIdx++) {
        costMatrix[pIdx][gIdx] = calculatePairwiseCost(
          predictedBboxes[pIdx],
          predictedLogits[pIdx],
          predictedEmbeddings[pIdx], // Pass predicted embedding
          gtObjects[gIdx]['bbox'] as List<double>,
          gtObjects[gIdx]['class_id'] as int,
          gtObjects[gIdx]['embedding']
              as List<double>, // Pass ground truth embedding
          numIdentities,
          embeddingDim,
        );
      }
    }

    // 3. Perform Bipartite Matching (Conceptual Hungarian Algorithm)
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
      final ValueVector currentPredictedEmbedding =
          predictedEmbeddings[predIdx]; // NEW
      final List<double> currentGtBboxCoords =
          gtObjects[gtIdx]['bbox'] as List<double>;
      final int currentGtClassId = gtObjects[gtIdx]['class_id'] as int;
      final List<double> currentGtEmbedding =
          gtObjects[gtIdx]['embedding'] as List<double>; // NEW

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
        numIdentities + 1, // numIdentities classes + 1 for background
        (i) => Value(i == currentGtClassId ? 1.0 : 0.0),
      ));
      final classLoss =
          currentPredictedLogits.softmax().crossEntropy(gtClassVector);

      // NEW: Embedding Loss (L1 Loss for recognition)
      Value embeddingLoss = Value(0.0);
      for (int i = 0; i < embeddingDim; i++) {
        embeddingLoss +=
            (currentPredictedEmbedding.values[i] - Value(currentGtEmbedding[i]))
                .abs();
      }
      embeddingLoss = embeddingLoss / Value(embeddingDim.toDouble());

      totalLoss += bboxLoss + classLoss + embeddingLoss; // Add embedding loss
    }

    // Loss for unmatched predicted objects (they should predict background)
    for (int pIdx = 0; pIdx < numQueries; pIdx++) {
      if (!matchedPredIndices.contains(pIdx)) {
        final ValueVector currentPredictedLogits = predictedLogits[pIdx];
        // Target is background class (numIdentities is the background ID)
        final gtBackgroundClassVector = ValueVector(List.generate(
          numIdentities + 1,
          (i) => Value(i == numIdentities ? 1.0 : 0.0),
        ));
        final backgroundClassLoss = currentPredictedLogits
            .softmax()
            .crossEntropy(gtBackgroundClassVector);
        totalLoss += backgroundClassLoss;
        // No bounding box loss or embedding loss for background predictions
      }
    }

    // 4. Backward pass and optimization step
    faceDetectorRecognizer.zeroGrad(); // Clear gradients
    totalLoss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 20 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Total Loss: ${totalLoss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Face Detector and Recognizer training complete.");

  // --- Inference Example ---
  print("\n--- Face Detector and Recognizer Inference ---");
  final List<double> newDummyImageData = List.generate(
      totalPixels, (i) => random.nextDouble()); // A new random image

  print(
      "New Dummy Image Data created (first 10 values): ${newDummyImageData.sublist(0, 10).map((v) => v.toStringAsFixed(2)).toList()}...");

  final Map<String, List<ValueVector>> inferencePredictions =
      faceDetectorRecognizer.forward(newDummyImageData);
  final List<ValueVector> inferredBboxes = inferencePredictions['boxes']!;
  final List<ValueVector> inferredLogits = inferencePredictions['logits']!;
  final List<ValueVector> inferredEmbeddings =
      inferencePredictions['embeddings']!; // NEW

  // --- Simulate a database of known face embeddings ---
  // In a real system, these would be pre-computed embeddings of known individuals.
  final Map<int, List<double>> knownFaceDatabase = {};
  for (int i = 0; i < numIdentities; i++) {
    // Generate a unique (but random for this demo) embedding for each identity
    knownFaceDatabase[i] =
        List.generate(embeddingDim, (j) => random.nextDouble() * 2 - 1);
  }
  print("\nSimulated Known Face Database (first 3 values of each embedding):");
  knownFaceDatabase.forEach((id, emb) {
    print(
        "  Identity $id: ${emb.sublist(0, 3).map((v) => v.toStringAsFixed(2)).toList()}...");
  });

  print("\nInferred Faces:");
  for (int q = 0; q < numQueries; q++) {
    final ValueVector currentInferredBbox = inferredBboxes[q];
    final ValueVector currentInferredLogits = inferredLogits[q];
    final ValueVector currentInferredProbs = currentInferredLogits.softmax();
    final ValueVector currentInferredEmbedding = inferredEmbeddings[q]; // NEW

    // Find the predicted class (identity or background)
    double maxProb = -1.0;
    int predictedClass = -1;
    for (int i = 0; i < currentInferredProbs.values.length; i++) {
      if (currentInferredProbs.values[i].data > maxProb) {
        maxProb = currentInferredProbs.values[i].data;
        predictedClass = i;
      }
    }

    // If a face is detected (not background and high confidence)
    if (predictedClass != numIdentities && maxProb > 0.5) {
      print("  Predicted Face ${q + 1}:");
      print(
          "    Bbox: ${currentInferredBbox.values.map((v) => v.data.toStringAsFixed(4)).toList()}");
      print(
          "    Detection Class: $predictedClass (Prob: ${maxProb.toStringAsFixed(4)})");
      print(
          "    Embedding (first 3): ${currentInferredEmbedding.values.map((v) => v.data.toStringAsFixed(4)).toList().sublist(0, 3)}...");

      // --- Face Recognition Logic ---
      // Compare the inferred embedding to the known face database
      double minDistance = double.infinity;
      int recognizedIdentity = -1;

      knownFaceDatabase.forEach((identityId, knownEmbedding) {
        double currentDistance = 0.0;
        for (int i = 0; i < embeddingDim; i++) {
          currentDistance +=
              (currentInferredEmbedding.values[i].data - knownEmbedding[i])
                  .abs(); // L1 distance
        }
        currentDistance /= embeddingDim; // Average distance

        if (currentDistance < minDistance) {
          minDistance = currentDistance;
          recognizedIdentity = identityId;
        }
      });

      // Set a threshold for recognition
      const double recognitionThreshold =
          0.5; // Example threshold (needs tuning)
      if (minDistance < recognitionThreshold) {
        print(
            "    Recognized as Identity: $recognizedIdentity (Distance: ${minDistance.toStringAsFixed(4)})");
      } else {
        print(
            "    Identity: Unknown (Closest: $recognizedIdentity, Distance: ${minDistance.toStringAsFixed(4)})");
      }
    } else if (predictedClass == numIdentities && maxProb > 0.5) {
      print(
          "  Predicted Object ${q + 1}: Background (Prob: ${maxProb.toStringAsFixed(4)})");
    } else {
      print(
          "  Predicted Object ${q + 1}: Low confidence prediction (Class: $predictedClass, Prob: ${maxProb.toStringAsFixed(4)}) - Likely background or noise");
    }
  }

  print(
      "\nNote: This example demonstrates face detection and recognition conceptually. "
      "Real-world systems use large, diverse datasets, advanced metric learning losses "
      "(e.g., Triplet Loss, ArcFace), more robust matching algorithms (Hungarian), "
      "and sophisticated post-processing (NMS) for accurate results.");
}
