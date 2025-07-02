// file: example_audio_video.dart

import 'dart:math';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'audio_transformer.dart'; // Import your AudioTransformer
import 'video_transformer.dart'; // Import your VideoTransformer

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
  final random = Random();

  // --- Audio Transformer Example ---
  print("--- Audio Transformer Example ---");

  final audioFeatureDim = 40; // e.g., 40 MFCCs
  final maxAudioSequenceLength = 100; // e.g., 10 seconds of 100ms frames
  final audioEmbedSize = 64;
  final audioNumClasses = 5; // e.g., classify spoken digits

  final audioModel = AudioTransformer(
    featureDim: audioFeatureDim,
    embedSize: audioEmbedSize,
    maxAudioSequenceLength: maxAudioSequenceLength,
    numClasses: audioNumClasses,
    numLayers: 1, // Keep small for example
    numHeads: 2, // Keep small for example
  );

  final audioOptimizer = SGD(audioModel.parameters(), 0.01);

  // Create dummy audio feature sequence
  final List<ValueVector> dummyAudioFeatures = List.generate(
      50, // Example sequence length
      (i) => ValueVector.fromDoubleList(
          List.generate(audioFeatureDim, (j) => random.nextDouble())));
  final int dummyAudioTargetClass = random.nextInt(audioNumClasses);

  print("Dummy Audio Sequence Length: ${dummyAudioFeatures.length}");
  print("Dummy Audio Target Class: $dummyAudioTargetClass");

  // Simplified Audio Training Loop
  print("\nTraining Audio Transformer...");
  for (int epoch = 0; epoch < 20; epoch++) {
    final logits = audioModel.forward(dummyAudioFeatures);
    final targetVector = ValueVector(List.generate(
      audioNumClasses,
      (i) => Value(i == dummyAudioTargetClass ? 1.0 : 0.0),
    ));
    final logitsVector = ValueVector(logits);
    final loss = logitsVector.softmax().crossEntropy(targetVector);

    audioModel.zeroGrad();
    loss.backward();
    audioOptimizer.step();

    if (epoch % 5 == 0 || epoch == 19) {
      print("Audio Epoch $epoch | Loss: ${loss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Audio Transformer training complete.");

  // Audio Inference
  final List<ValueVector> newDummyAudioFeatures = List.generate(
      60, // Different length for inference
      (i) => ValueVector.fromDoubleList(
          List.generate(audioFeatureDim, (j) => random.nextDouble())));
  final audioInferenceLogits = audioModel.forward(newDummyAudioFeatures);
  final audioPredictedProbs = ValueVector(audioInferenceLogits).softmax();
  int audioPredictedClass = audioPredictedProbs.values
      .asMap()
      .entries
      .reduce((a, b) => a.value.data > b.value.data ? a : b)
      .key;
  print(
      "Audio Predicted Class: $audioPredictedClass (Prob: ${audioPredictedProbs.values[audioPredictedClass].data.toStringAsFixed(4)})");

  // --- Video Transformer Example ---
  print("\n--- Video Transformer Example ---");

  final frameEmbedDim = 128; // e.g., embedding from a frame-level CNN/ViT
  final maxVideoSequenceLength = 30; // e.g., 30 frames/clips
  final videoEmbedSize = 128; // Matching frameEmbedDim
  final videoNumClasses = 10; // e.g., action classes

  final videoModel = VideoTransformer(
    frameEmbedDim: frameEmbedDim,
    embedSize: videoEmbedSize,
    maxVideoSequenceLength: maxVideoSequenceLength,
    numClasses: videoNumClasses,
    numLayers: 1, // Keep small for example
    numHeads: 2, // Keep small for example
  );

  final videoOptimizer = SGD(videoModel.parameters(), 0.01);

  // Create dummy video frame embedding sequence
  final List<ValueVector> dummyVideoEmbeddings = List.generate(
      20, // Example video length
      (i) => ValueVector.fromDoubleList(
          List.generate(frameEmbedDim, (j) => random.nextDouble())));
  final int dummyVideoTargetClass = random.nextInt(videoNumClasses);

  print("Dummy Video Sequence Length: ${dummyVideoEmbeddings.length}");
  print("Dummy Video Target Class: $dummyVideoTargetClass");

  // Simplified Video Training Loop
  print("\nTraining Video Transformer...");
  for (int epoch = 0; epoch < 20; epoch++) {
    final logits = videoModel.forward(dummyVideoEmbeddings);
    final targetVector = ValueVector(List.generate(
      videoNumClasses,
      (i) => Value(i == dummyVideoTargetClass ? 1.0 : 0.0),
    ));
    final logitsVector = ValueVector(logits);
    final loss = logitsVector.softmax().crossEntropy(targetVector);

    videoModel.zeroGrad();
    loss.backward();
    videoOptimizer.step();

    if (epoch % 5 == 0 || epoch == 19) {
      print("Video Epoch $epoch | Loss: ${loss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Video Transformer training complete.");

  // Video Inference
  final List<ValueVector> newDummyVideoEmbeddings = List.generate(
      25, // Different length for inference
      (i) => ValueVector.fromDoubleList(
          List.generate(frameEmbedDim, (j) => random.nextDouble())));
  final videoInferenceLogits = videoModel.forward(newDummyVideoEmbeddings);
  final videoPredictedProbs = ValueVector(videoInferenceLogits).softmax();
  int videoPredictedClass = videoPredictedProbs.values
      .asMap()
      .entries
      .reduce((a, b) => a.value.data > b.value.data ? a : b)
      .key;
  print(
      "Video Predicted Class: $videoPredictedClass (Prob: ${videoPredictedProbs.values[videoPredictedClass].data.toStringAsFixed(4)})");

  print(
      "\nNote: The input data for these examples are dummy `ValueVector` sequences. "
      "Real-world audio/video processing (e.g., loading files, extracting features/frames) "
      "would require specialized Dart libraries or external tools.");
}
