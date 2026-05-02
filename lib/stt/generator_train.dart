import 'dart:math' as math;

import '../transformer/transformer_decoder.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/multi_modal_generator.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../nn/value_vector.dart';
import '../nn/value.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';

// class SGD {
//   final List<Value> parameters;
//   final double learningRate;
//   SGD(this.parameters, this.learningRate);

//   void step() {
//     for (final p in parameters) {
//       p.data -= learningRate * p.grad;
//     }
//   }

//   void zeroGrad() {
//     for (final p in parameters) {
//       p.grad = 0.0;
//     }
//   }
// }

void main(List<String> args) async {
  print("--- Multimodal Training: Overfitting a Single Sample ---");

  String wavPath = args.isNotEmpty ? args[0] : "test.wav";

  // --- Parameters ---
  const int commonEmbedSize = 32;
  const int vocabSize = 50;
  const int maxTextLen = 12;

  // --- Model Instantiation ---
  final audioModel = AudioTransformer(
    featureDim: 32,
    embedSize: commonEmbedSize,
    maxAudioSequenceLength: 64,
    numClasses: 1,
    numLayers: 1,
    numHeads: 2,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: 64, // Must match the length of individual videoFrames
    embedSize: commonEmbedSize,
    maxVideoSequenceLength: 20,
    numClasses: 1,
    numLayers: 1,
    numHeads: 2,
  );

  final decoder = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: commonEmbedSize,
    encoderEmbedSize: commonEmbedSize,
    blockSize: maxTextLen,
  );

  final generator = MultimodalGenerator(
    audioEncoder: audioModel,
    videoEncoder: videoModel,
    decoder: decoder,
  );

  // --- Data Preparation ---

  // 1. Audio
  final rawSpectrogram =
      await melSpectrogram(wavPath, sampleRate: 16000, nMels: 32);
  final audioInput = MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: 64);

  // 2. Video: Define 20 frames of 64 features each
  final List<List<double>> videoFrames = List.generate(
      20, (i) => List.generate(64, (j) => math.Random().nextDouble()));

  // Use your buffer to convert these into List<ValueVector>
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  // --- Training Logic ---
  final List<int> targetTokens = [
    0,
    5,
    12,
    3,
    1
  ]; // <SOS>, The, Transformer, Nominal, <EOS>
  const int epochs = 50;
  const double learningRate =
      0.05; // Slightly higher for faster overfitting on CPU

  for (int epoch = 1; epoch <= epochs; epoch++) {
    for (var p in generator.parameters()) p.grad = 0.0;

    final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
    final expectedTargets = targetTokens.sublist(1);

    // Pass videoInput here instead of dummyVideo
    final List<ValueVector> logits =
        generator.forward(audioInput, videoInput, inputTokens);

    Value totalLoss = Value(0.0);
    for (int i = 0; i < logits.length; i++) {
      final targetVector = List.generate(
          vocabSize, (idx) => idx == expectedTargets[i] ? 1.0 : 0.0);
      totalLoss +=
          logits[i].crossEntropy(ValueVector.fromDoubleList(targetVector));
    }

    totalLoss.backward();

    for (var p in generator.parameters()) {
      p.data -= learningRate * p.grad;
    }

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${totalLoss.data.toStringAsFixed(6)}");
    }
  }

  print("\n--- Overfitting Complete ---");
  verifyInference(generator, audioInput, videoInput);
}

void verifyInference(
    MultimodalGenerator gen, List<ValueVector> audio, List<ValueVector> video) {
  List<int> current = [0];
  for (int i = 0; i < 5; i++) {
    final logits = gen.forward(audio, video, current);
    final probs = logits.last.softmax();

    int nextId = 0;
    double maxP = -1.0;
    for (int j = 0; j < probs.values.length; j++) {
      if (probs.values[j].data > maxP) {
        maxP = probs.values[j].data;
        nextId = j;
      }
    }
    current.add(nextId);
    if (nextId == 1) break;
  }
  print("Model Prediction: $current");
}
