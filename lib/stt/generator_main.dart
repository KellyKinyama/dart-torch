import 'dart:math' as math;
import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer/example_aft_full_cycle3.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../transformer_misc/multi_modal_generator.dart';
import '../transformer/transformer_decoder.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';

// 1. Place the SGD class at the top level or import it
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

  // --- 2. Configuration (Match your Inference parameters) ---
  const int commonEmbedSize = 32;
  const int vocabSize = 50;
  const int maxTextLen = 12;
  const int audioMels = 32;
  String wavPath = args.isNotEmpty ? args[0] : "test.wav";

  // --- 3. Model Instantiation ---
  // --- 3. Model Instantiation ---
  final audioModel = AudioTransformer(
    featureDim: audioMels,
    embedSize: commonEmbedSize,
    maxAudioSequenceLength: 64,
    numClasses: 1, // Add this: required by constructor but unused in generation
    numLayers: 1, // Optional: check if your constructor requires these too
    numHeads: 2,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: 64,
    embedSize: commonEmbedSize,
    maxVideoSequenceLength: 20,
    numClasses: 1, // Add this
    numLayers: 1,
    numHeads: 2,
  );

  final decoder = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: commonEmbedSize,
    encoderEmbedSize: commonEmbedSize,
    blockSize: maxTextLen,
    numLayers: 1, // Add this
    numHeads: 2, // Add this
  );

  // final decoder = TransformerDecoder(
  //   vocabSize: vocabSize,
  //   embedSize: commonEmbedSize,
  //   encoderEmbedSize: commonEmbedSize,
  //   blockSize: maxTextLen,
  // );

  final generator = MultimodalGenerator(
    audioEncoder: audioModel,
    videoEncoder: videoModel,
    decoder: decoder,
  );

  // --- 4. Data Preparation ---
  final rawSpectrogram =
      await melSpectrogram(wavPath, sampleRate: 16000, nMels: audioMels);
  final audioInput = MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: 64);

  // Define synthetic video frames
  final List<List<double>> videoFrames = List.generate(
      20, (i) => List.generate(64, (j) => math.Random().nextDouble()));
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  // Define Target Sequence: <SOS>, The, Transformer, Nominal, <EOS>
  final List<int> targetTokens = [0, 5, 12, 3, 1];
  final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
  final expectedTargets = targetTokens.sublist(1);

  // --- 5. The Stabilized Training Loop ---
  final optimizer = SGD(generator.parameters(), 0.001);

  for (int epoch = 1; epoch <= 100; epoch++) {
    optimizer.zeroGrad();

    // Forward pass
    final List<ValueVector> logits =
        generator.forward(audioInput, videoInput, inputTokens);

    Value totalLoss = Value(0.0);
    for (int i = 0; i < logits.length; i++) {
      final targetVector = ValueVector.fromDoubleList(List.generate(
          vocabSize, (idx) => idx == expectedTargets[i] ? 1.0 : 0.0));

      // Get stable probabilities and calculate loss
      final probs = logits[i].softmax();
      totalLoss += probs.crossEntropy(targetVector);
    }

    // Normalize loss by sequence length
    final normalizedLoss = totalLoss / Value(logits.length.toDouble());

    normalizedLoss.backward();

    // Gradient Clipping to prevent NaN
    // for (var p in generator.parameters()) {
    //   if (p.grad > 1.0) p.grad = 1.0;
    //   if (p.grad < -1.0) p.grad = -1.0;
    // }

    optimizer.step();

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${normalizedLoss.data.toStringAsFixed(6)}");
    }
  }

  print("\n--- Overfitting Complete ---");

  // Verification check
  verifyInference(generator, audioInput, videoInput);
}

void verifyInference(
    MultimodalGenerator gen, List<ValueVector> audio, List<ValueVector> video) {
  List<int> current = [0];
  for (int i = 0; i < 6; i++) {
    final logits = gen.forward(audio, video, current);
    final nextId = logits.last
        .softmax()
        .values
        .asMap()
        .entries
        .reduce((a, b) => a.value.data > b.value.data ? a : b)
        .key;
    current.add(nextId);
    if (nextId == 1) break;
  }
  print("Predicted Sequence: $current");
}
