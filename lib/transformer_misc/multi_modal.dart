// file: example_audio_video.dart

import 'dart:math';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'audio_transformer.dart'; // Import your AudioTransformer
import 'video_transformer.dart'; // Import your VideoTransformer
import '/nn/module.dart'; // Import Module for MultimodalTransformer
import '/nn/layer.dart'; // Import Layer for the joint MLP head

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

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

/// A Multimodal Transformer that combines audio and video features for joint classification.
///
/// This class orchestrates the forward passes of individual audio and video
/// transformers, concatenates their pooled features, and then passes them
/// through a shared MLP head for multimodal predictions.
class MultimodalTransformer extends Module {
  final AudioTransformer audioModel;
  final VideoTransformer videoModel;
  final Layer multimodalMlpHead;
  final int
      multimodalNumClasses; // Number of output classes for the combined task

  MultimodalTransformer({
    required this.audioModel,
    required this.videoModel,
    required this.multimodalNumClasses,
  }) : multimodalMlpHead = Layer.fromNeurons(
          audioModel.embedSize +
              videoModel.embedSize, // Sum of feature dimensions
          multimodalNumClasses,
        );

  /// The forward pass for the Multimodal Transformer.
  ///
  /// Takes raw audio features and video embeddings, processes them independently,
  /// and then combines their outputs for a joint prediction.
  List<Value> forward(
      List<ValueVector> audioFeatures, List<ValueVector> videoEmbeddings) {
    // Process audio features using the AudioTransformer's internal logic
    // We need the pooled feature *before* the final MLP head of the AudioTransformer.
    if (audioFeatures.length > audioModel.maxAudioSequenceLength) {
      throw ArgumentError(
          "Audio sequence length (${audioFeatures.length}) exceeds "
          "maxAudioSequenceLength (${audioModel.maxAudioSequenceLength}).");
    }
    final embeddedAudioFeatures = audioFeatures
        .map((f) => audioModel.featureProjection.forward(f))
        .toList();
    final audioSequenceWithPositionalEmbeddings =
        List.generate(embeddedAudioFeatures.length, (i) {
      return embeddedAudioFeatures[i] + audioModel.positionEmbeddings[i];
    });
    final encodedAudioFeatures = audioModel.transformerEncoder
        .forwardEmbeddings(audioSequenceWithPositionalEmbeddings);
    ValueVector pooledAudioFeature;
    if (encodedAudioFeatures.isEmpty) {
      pooledAudioFeature =
          ValueVector(List.filled(audioModel.embedSize, Value(0.0)));
    } else {
      pooledAudioFeature = encodedAudioFeatures.reduce((a, b) => a + b) /
          Value(encodedAudioFeatures.length.toDouble());
    }

    // Process video features using the VideoTransformer's internal logic
    // We need the pooled feature *before* the final MLP head of the VideoTransformer.
    if (videoEmbeddings.length > videoModel.maxVideoSequenceLength) {
      throw ArgumentError(
          "Video sequence length (${videoEmbeddings.length}) exceeds "
          "maxVideoSequenceLength (${videoModel.maxVideoSequenceLength}).");
    }
    final projectedVideoEmbeddings = videoModel.frameProjection != null
        ? videoEmbeddings
            .map((e) => videoModel.frameProjection!.forward(e))
            .toList()
        : videoEmbeddings;
    final videoSequenceWithPositionalEmbeddings =
        List.generate(projectedVideoEmbeddings.length, (i) {
      return projectedVideoEmbeddings[i] + videoModel.positionEmbeddings[i];
    });
    final encodedVideoFeatures = videoModel.transformerEncoder
        .forwardEmbeddings(videoSequenceWithPositionalEmbeddings);
    ValueVector pooledVideoFeature;
    if (encodedVideoFeatures.isEmpty) {
      pooledVideoFeature =
          ValueVector(List.filled(videoModel.embedSize, Value(0.0)));
    } else {
      pooledVideoFeature = encodedVideoFeatures.reduce((a, b) => a + b) /
          Value(encodedVideoFeatures.length.toDouble());
    }

    // Concatenate the pooled features from both modalities
    final combinedFeature = ValueVector(
        [...pooledAudioFeature.values, ...pooledVideoFeature.values]);

    // Pass the combined feature through the multimodal classification head
    final logits = multimodalMlpHead.forward(combinedFeature);

    return logits
        .values; // Return a list of Value objects for a single prediction
  }

  @override
  List<Value> parameters() {
    return [
      ...audioModel.parameters(),
      ...videoModel.parameters(),
      ...multimodalMlpHead.parameters(),
    ];
  }

  @override
  void zeroGrad() {
    audioModel.zeroGrad(); // Clear gradients for audio model
    videoModel.zeroGrad(); // Clear gradients for video model
    multimodalMlpHead.zeroGrad(); // Clear gradients for multimodal head
  }
}

void main() {
  final random = Random();

  // --- Multimodal Transformer Example ---
  print("--- Multimodal Transformer Example (Audio + Video Fusion) ---");

  // Audio model parameters (from audio_transformer.dart)
  final audioFeatureDim = 40;
  final maxAudioSequenceLength = 100;
  final audioEmbedSize = 64;
  final audioNumClasses = 5; // Individual audio classification classes

  // Video model parameters (from video_transformer.dart)
  final frameEmbedDim = 128;
  final maxVideoSequenceLength = 30;
  final videoEmbedSize = 128;
  final videoNumClasses = 10; // Individual video classification classes

  // Multimodal task parameters
  final multimodalNumClasses =
      7; // Example: 7 classes for combined audio-video events

  // Instantiate individual Audio and Video Transformers
  final audioModel = AudioTransformer(
    featureDim: audioFeatureDim,
    embedSize: audioEmbedSize,
    maxAudioSequenceLength: maxAudioSequenceLength,
    numClasses:
        audioNumClasses, // These classes are for the *individual* audio task
    numLayers: 1,
    numHeads: 2,
  );
  final videoModel = VideoTransformer(
    frameEmbedDim: frameEmbedDim,
    embedSize: videoEmbedSize,
    maxVideoSequenceLength: maxVideoSequenceLength,
    numClasses:
        videoNumClasses, // These classes are for the *individual* video task
    numLayers: 1,
    numHeads: 2,
  );

  // Instantiate the Multimodal Transformer
  final multimodalModel = MultimodalTransformer(
    audioModel: audioModel,
    videoModel: videoModel,
    multimodalNumClasses: multimodalNumClasses,
  );

  final multimodalOptimizer = SGD(multimodalModel.parameters(), 0.01);

  // Create dummy multimodal input data
  final List<ValueVector> dummyAudioFeatures = List.generate(
      50, // Example sequence length
      (i) => ValueVector.fromDoubleList(
          List.generate(audioFeatureDim, (j) => random.nextDouble())));
  final List<ValueVector> dummyVideoEmbeddings = List.generate(
      20, // Example video length
      (i) => ValueVector.fromDoubleList(
          List.generate(frameEmbedDim, (j) => random.nextDouble())));
  final int dummyMultimodalTargetClass = random.nextInt(multimodalNumClasses);

  print("Dummy Audio Sequence Length: ${dummyAudioFeatures.length}");
  print("Dummy Video Sequence Length: ${dummyVideoEmbeddings.length}");
  print("Dummy Multimodal Target Class: $dummyMultimodalTargetClass");

  // Simplified Multimodal Training Loop
  print("\nTraining Multimodal Transformer...");
  for (int epoch = 0; epoch < 50; epoch++) {
    // Increased epochs for multimodal example
    final logits =
        multimodalModel.forward(dummyAudioFeatures, dummyVideoEmbeddings);
    final targetVector = ValueVector(List.generate(
      multimodalNumClasses,
      (i) => Value(i == dummyMultimodalTargetClass ? 1.0 : 0.0),
    ));
    final logitsVector = ValueVector(logits);
    final loss = logitsVector.softmax().crossEntropy(targetVector);

    multimodalModel.zeroGrad();
    loss.backward();
    multimodalOptimizer.step();

    if (epoch % 1 == 0 || epoch == 49) {
      print("Multimodal Epoch $epoch | Loss: ${loss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Multimodal Transformer training complete.");

  // Multimodal Inference
  print("\n--- Multimodal Transformer Inference ---");
  final List<ValueVector> newDummyAudioFeatures = List.generate(
      60, // Different length for inference
      (i) => ValueVector.fromDoubleList(
          List.generate(audioFeatureDim, (j) => random.nextDouble())));
  final List<ValueVector> newDummyVideoEmbeddings = List.generate(
      25, // Different length for inference
      (i) => ValueVector.fromDoubleList(
          List.generate(frameEmbedDim, (j) => random.nextDouble())));

  final multimodalInferenceLogits =
      multimodalModel.forward(newDummyAudioFeatures, newDummyVideoEmbeddings);
  final multimodalPredictedProbs =
      ValueVector(multimodalInferenceLogits).softmax();
  int multimodalPredictedClass = multimodalPredictedProbs.values
      .asMap()
      .entries
      .reduce((a, b) => a.value.data > b.value.data ? a : b)
      .key;
  print(
      "Multimodal Predicted Class: $multimodalPredictedClass (Prob: ${multimodalPredictedProbs.values[multimodalPredictedClass].data.toStringAsFixed(4)})");

  print("\nNote: This example demonstrates a basic intermediate fusion. "
      "Real-world multimodal models often involve more complex fusion mechanisms (e.g., cross-attention) "
      "and larger datasets for effective learning.");
}
