import 'dart:math'; // Fixes 'Random' not defined
import 'dart:io';

// Core NN Imports (Adjust paths based on your project structure)
import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../nn/module.dart';

// Transformer & Utility Imports
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/multi_modal.dart';
import '../transformer_misc/video_transformer.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';

Future<void> runInference(
    String wavPath, List<List<double>> videoFrames) async {

  print("--- Multimodal Transformer Inference (Audio + Video Fusion) ---");

  // --- 1. Parameters ---
  // FIX: audioFeatureDim MUST match the nMels you pass to melSpectrogram
  final audioFeatureDim = 64;
  final maxAudioSequenceLength = 100;
  final audioEmbedSize = 64;
  final audioNumClasses = 5;

  final frameEmbedDim = 128;
  final maxVideoSequenceLength = 30;
  final videoEmbedSize = 128;
  final videoNumClasses = 10;

  final multimodalNumClasses = 7;

  // --- 2. Model Instantiation ---
  final audioModel = AudioTransformer(
    featureDim: audioFeatureDim,
    embedSize: audioEmbedSize,
    maxAudioSequenceLength: maxAudioSequenceLength,
    numClasses: audioNumClasses,
    numLayers: 2,
    numHeads: 4,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: frameEmbedDim,
    embedSize: videoEmbedSize,
    maxVideoSequenceLength: maxVideoSequenceLength,
    numClasses: videoNumClasses,
    numLayers: 2,
    numHeads: 4,
  );

  final multimodalModel = MultimodalTransformer(
    audioModel: audioModel,
    videoModel: videoModel,
    multimodalNumClasses: multimodalNumClasses,
  );

  // --- 3. Data Processing ---

  // Generate Spectrogram
  // nMels now correctly matches audioFeatureDim (64)
  final rawSpectrogram = await melSpectrogram(
    wavPath,
    sampleRate: 22050,
    nMels: audioFeatureDim,
  );

  // Use Buffer to convert and normalize data
  final audioInput = MultimodalBuffer.prepareAudio(
    rawSpectrogram,
    maxLen: multimodalModel.audioModel.maxAudioSequenceLength,
    topDb: 80.0,
  );

  final videoInput = MultimodalBuffer.prepareVideo(
    videoFrames,
    maxLen: multimodalModel.videoModel.maxVideoSequenceLength,
  );

  // --- 4. Forward Pass ---
  // logits is a List<Value>
  final logits = multimodalModel.forward(audioInput, videoInput);

  // --- 5. Post-processing ---
  final probs = ValueVector(logits).softmax();

  // Extract double values from the Value objects for clean printing
  final doubleProbs =
      probs.values.map((v) => v.data.toStringAsFixed(4)).toList();
  print("Prediction Probabilities: $doubleProbs");

  // Calculate predicted class (Argmax)
  int predictedClass = 0;
  double maxProb = -1.0;
  for (int i = 0; i < probs.values.length; i++) {
    if (probs.values[i].data > maxProb) {
      maxProb = probs.values[i].data;
      predictedClass = i;
    }
  }
  print(
      "Predicted Class: $predictedClass with probability ${maxProb.toStringAsFixed(4)}");
}
