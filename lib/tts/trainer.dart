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
  const int leanEmbedSize = 32; // Half the size = faster math
  const int audioMels = 32;
  const int maxTextLen = 30;
  const int maxAudioLen = 64; // Reduced window for faster processing
  const int sampleRate = 16000;

  // 2. Optimized Data Prep
  final targetSpectrogram =
      await melSpectrogram("output.wav", nMels: audioMels);

  final List<ValueVector> targetMels = targetSpectrogram
      .take(maxAudioLen) // Only take what we need
      .map((m) => ValueVector.fromDoubleList(m.toList()))
      .toList();

  String labelText = "IF HE'D RUN OUT OF";
  final List<int> inputTokens = tokenizer.encode(labelText, maxLen: maxTextLen);

  // 3. Lean Model Setup
  final ttsDecoder = TransformerDecoder(
    vocabSize: audioMels,
    embedSize: leanEmbedSize,
    encoderEmbedSize: leanEmbedSize,
    blockSize: maxAudioLen,
    numLayers: 1, // Single layer for massive speedup
    numHeads: 2, // Fewer heads = less context graph overhead
  );

  final optimizer =
      SGD(ttsDecoder.parameters(), 0.01); // Increased LR for faster convergence
  final List<int> timeIndices =
      List.generate(maxAudioLen, (i) => i % audioMels);

  // 4. Training Loop (Reduced epochs because it learns faster)
  for (int epoch = 1; epoch <= 150; epoch++) {
    optimizer.zeroGrad();

    // Use current embedding table for dummy context
    final List<ValueVector> dummyEncoderOutput = inputTokens.map((token) {
      return ttsDecoder
          .positionEmbeddings[token % ttsDecoder.positionEmbeddings.length];
    }).toList();

    final List<ValueVector> predictedMels =
        ttsDecoder.forward(timeIndices, dummyEncoderOutput);

    Value totalMseLoss = Value(0.0);
    int compareLen = math.min(predictedMels.length, targetMels.length);

    // Optimized Loss loop
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

    // Clipping stays for stability
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
  // Proceed to Griffin-Lim generation as before...

  // 5. Audio Generation Phase
  // ---------------------------------------------------------

  // Final inference pass
  final List<ValueVector> finalContext = inputTokens.map((token) {
    return ttsDecoder
        .positionEmbeddings[token % ttsDecoder.positionEmbeddings.length];
  }).toList();

  final List<ValueVector> finalMels =
      ttsDecoder.forward(timeIndices, finalContext);

  // Convert tensors back to raw magnitudes
  List<Float64List> magnitudes = finalMels.map((vec) {
    return Float64List.fromList(vec.values.map((v) => v.data).toList());
  }).toList();

  // Griffin-Lim reconstruction (the "opposite" of Mel extraction)
  final gl = GriffinLimGenerator(
    iterations: 60, // Lowered slightly for faster CPU turnaround
    frameSize: 1024,
    hopSize: 256,
  );

  final Uint8List pcmBytes = gl.generateWav(magnitudes, sampleRate);

  // Final Encoding to .wav
  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final outputFile = File("lean_generated_output.wav");
  encoder.encode(outputFile, pcmBytes);

  print("Success! Play the file here: ${outputFile.absolute.path}");
}
