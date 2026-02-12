// file: main_face_recognition.dart

import 'dart:math';
import 'dart:math' as math;
import '/nn/value.dart';
import '/nn/value_vector.dart';
// import '/nn/module.dart'; // Import Module for SGD to inherit from if it's a Module
import 'aft_vit_face_embedding_model.dart'; // Our modified ViT for embeddings
import 'triplet_loss.dart'; // Our Triplet Loss implementation

// Re-using a simple SGD optimizer (can be a standalone class or part of a utilities file)
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

  // @override
  // List<Value> parameters() =>
  //     []; // SGD itself doesn't have learnable parameters
}

// Helper to generate dummy image data
List<double> generateDummyImageData(int size, int channels) {
  final Random random = Random();
  return List.generate(size * size * channels, (i) => random.nextDouble());
}

void main() {
  print("--- ViT-based Face Recognition Example with Triplet Loss ---");

  // Model parameters
  final imageSize = 32; // Example: Small 32x32 image
  final patchSize = 8; // Patches will be 8x8 pixels
  final numChannels = 3; // RGB image
  final embedSize =
      128; // Transformer embedding dimension (also our face embedding size)
  final numLayers = 2; // Small number of layers for quick execution
  final numHeads = 4; // Number of attention heads

  print("Model Configuration:");
  print("  Image Size: $imageSize x $imageSize");
  print("  Patch Size: $patchSize x $patchSize");
  print("  Embedding Size: $embedSize");

  // Instantiate the ViTFaceEmbeddingModel
  final faceEmbeddingModel = ViTFaceEmbeddingModel(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  // Instantiate the Triplet Loss
  final double margin = 0.5; // A common margin value
  final tripletLossFn = TripletLoss(margin: margin);

  // Optimizer
  final optimizer = SGD(faceEmbeddingModel.parameters(), 0.01);

  // --- Dummy Training Data (Anchor, Positive, Negative) ---
  // In a real scenario, you'd load actual face images.
  // Here, we simulate by generating random data.
  // For a proper triplet, anchor and positive are from the same person,
  // and negative is from a different person.

  // Simulate Anchor (Person A)
  final List<double> anchorImageData =
      generateDummyImageData(imageSize, numChannels);
  // Simulate Positive (another image of Person A)
  final List<double> positiveImageData =
      generateDummyImageData(imageSize, numChannels);
  // Simulate Negative (image of Person B)
  final List<double> negativeImageData =
      generateDummyImageData(imageSize, numChannels);

  print(
      "\nDummy Image Data generated for Anchor, Positive, and Negative samples.");

  // --- Training Loop ---
  final epochs = 15;
  print("\nTraining Face Embedding Model for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // 1. Forward pass: Get embeddings for anchor, positive, and negative
    final ValueVector anchorEmbedding =
        faceEmbeddingModel.forward(anchorImageData);
    final ValueVector positiveEmbedding =
        faceEmbeddingModel.forward(positiveImageData);
    final ValueVector negativeEmbedding =
        faceEmbeddingModel.forward(negativeImageData);

    // 2. Calculate Triplet Loss
    final Value loss = tripletLossFn.forward(
        anchorEmbedding, positiveEmbedding, negativeEmbedding);

    // 3. Backward pass and optimization step
    optimizer.zeroGrad(); // Clear gradients
    loss.backward(); // Compute gradients
    optimizer.step(); // Update parameters

    if (epoch % 1 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Triplet Loss: ${loss.data.toStringAsFixed(6)}");
    }
  }
  print("✅ Face Embedding Model training complete with Triplet Loss.");

  // --- Inference Example ---
  print("\n--- Face Embedding Model Inference ---");

  // Get embeddings for new dummy images
  final List<double> testImage1Data =
      generateDummyImageData(imageSize, numChannels); // Person X
  final List<double> testImage2Data = generateDummyImageData(
      imageSize, numChannels); // Person X (another image)
  final List<double> testImage3Data =
      generateDummyImageData(imageSize, numChannels); // Person Y

  final ValueVector embedding1 = faceEmbeddingModel.forward(testImage1Data);
  final ValueVector embedding2 = faceEmbeddingModel.forward(testImage2Data);
  final ValueVector embedding3 = faceEmbeddingModel.forward(testImage3Data);

  print("\nInferred Embeddings (first 5 values):");
  print(
      "  Embedding 1: ${embedding1.values.sublist(0, math.min(5, embedding1.values.length)).map((v) => v.data.toStringAsFixed(4)).toList()}...");
  print(
      "  Embedding 2: ${embedding2.values.sublist(0, math.min(5, embedding2.values.length)).map((v) => v.data.toStringAsFixed(4)).toList()}...");
  print(
      "  Embedding 3: ${embedding3.values.sublist(0, math.min(5, embedding3.values.length)).map((v) => v.data.toStringAsFixed(4)).toList()}...");

  // Calculate distances between embeddings
  final Value distance1_2 =
      TripletLoss.euclideanDistance(embedding1, embedding2);
  final Value distance1_3 =
      TripletLoss.euclideanDistance(embedding1, embedding3);

  print("\nDistances:");
  print(
      "  Distance between Embedding 1 and Embedding 2 (same person simulation): ${distance1_2.data.toStringAsFixed(4)}");
  print(
      "  Distance between Embedding 1 and Embedding 3 (different person simulation): ${distance1_3.data.toStringAsFixed(4)}");

  print(
      "\nExpected behavior: After training, 'Distance between Embedding 1 and Embedding 2' should ideally be smaller than 'Distance between Embedding 1 and Embedding 3'.");
  print(
      "Note: With random dummy data and a small model, significant convergence might not be observed without more extensive training and real data.");
}
