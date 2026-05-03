import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../stt/audio_to_spectogram/audio_spectrogram.dart';
import '../stt/tokenizer.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer/transformer_decoder.dart';
import 'griffin_lim_generator.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';
// import 'audio_to_spectogram/audio_spectrogram.dart';
// import 'tokenizer/english_character_tokenizer.dart';

// --- SGD remains top-level or imported ---
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

void main() async {
  print("--- LEAN TTS Overfitting: Rapid Learning Mode ---");

  final tokenizer = EnglishCharacterTokenizer();

  // 1. Lean Configuration
  const int leanEmbedSize = 32;
  const int audioMels = 32;
  const int maxTextLen = 30;
  const int maxAudioLen = 64;
  const int sampleRate = 16000;

  // 2. Optimized Data Prep
  // Ensure "output.wav" was saved correctly using the 16-bit Int16List fix
  final targetSpectrogram =
      await melSpectrogram("output.wav", nMels: audioMels);

  final List<ValueVector> targetMels = targetSpectrogram
      .take(maxAudioLen)
      .map((m) => ValueVector.fromDoubleList(m.toList()))
      .toList();

  String labelText = "IF HE'D RUN OUT OF";
  final List<int> inputTokens = tokenizer.encode(labelText, maxLen: maxTextLen);

  // 3. Lean Model Setup
  // final ttsDecoder = TransformerDecoder(
  //   vocabSize: audioMels,
  //   embedSize: leanEmbedSize,
  //   encoderEmbedSize: leanEmbedSize,
  //   blockSize: maxAudioLen,
  //   numLayers: 1,
  //   numHeads: 2,
  // );

  // 3. Lean Model Setup
  final ttsDecoder = TransformerDecoder(
    // FIX: vocabSize must be at least maxAudioLen to accommodate the timeIndices
    vocabSize: maxAudioLen,
    embedSize: leanEmbedSize,
    encoderEmbedSize: leanEmbedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final optimizer = SGD(ttsDecoder.parameters(), 0.01);

  // FIX: Continuous time indices (0, 1, 2...) instead of modulo
  final List<int> timeIndices = List.generate(maxAudioLen, (i) => i);

  // 4. Training Loop
  for (int epoch = 1; epoch <= 200; epoch++) {
    optimizer.zeroGrad();

    // FIX: Stabilized context mapping
    // We map the input text tokens to a fixed latent space for the decoder to reference
    final List<ValueVector> encoderOutput = inputTokens.map((token) {
      return ValueVector.fromDoubleList(List.generate(leanEmbedSize,
          (idx) => (token + idx) / (tokenizer.vocabSize + leanEmbedSize)));
    }).toList();

    final List<ValueVector> predictedMels =
        ttsDecoder.forward(timeIndices, encoderOutput);

    Value totalMseLoss = Value(0.0);
    int compareLen = math.min(predictedMels.length, targetMels.length);

    for (int i = 0; i < compareLen; i++) {
      final diff = predictedMels[i] - targetMels[i];
      final squaredDiff = diff.squared();
      for (var val in squaredDiff.values) {
        totalMseLoss += val;
      }
    }

    final normalizedLoss =
        totalMseLoss / Value(compareLen.toDouble() * audioMels);
    normalizedLoss.backward();

    // Gradient clipping for stability in the Artificial Intelligence & Machine Learning Department
    for (var p in ttsDecoder.parameters()) {
      p.grad = p.grad.clamp(-1.0, 1.0);
    }

    optimizer.step();

    if (epoch % 5 == 0 || epoch == 1) {
      print(
          "Epoch $epoch | MSE Loss: ${normalizedLoss.data.toStringAsFixed(8)}");
    }
  }

  print("\n--- Lean Overfit Complete ---");

  // 5. Audio Generation Phase
  // ---------------------------------------------------------

  final List<ValueVector> finalContext = inputTokens.map((token) {
    return ValueVector.fromDoubleList(List.generate(leanEmbedSize,
        (idx) => (token + idx) / (tokenizer.vocabSize + leanEmbedSize)));
  }).toList();

  final List<ValueVector> finalMels =
      ttsDecoder.forward(timeIndices, finalContext);

  // Convert predicted tensors back to Float64 magnitudes
  List<Float64List> magnitudes = finalMels.map((vec) {
    return Float64List.fromList(vec.values.map((v) => v.data).toList());
  }).toList();

  // Griffin-Lim reconstruction
  final gl = GriffinLimGenerator(
    iterations: 100, // Balanced for quality vs speed
    frameSize: 1024,
    hopSize: 256,
  );

  // Generate the raw PCM bytes
  final Uint8List pcmBytes = gl.generateWav(magnitudes, sampleRate);

  // Final Encoding to WAV
  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final outputFile = File("lean_generated_output.wav");
  encoder.encode(outputFile, pcmBytes);

  print("Success! Play the file here: ${outputFile.absolute.path}");
}
