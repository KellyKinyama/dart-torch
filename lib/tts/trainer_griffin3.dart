import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../stft_spectrogram.dart';
import '../stt/tokenizer.dart';
import '../transformer/transformer_decoder.dart';
import 'griffin_lim_generator.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

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
  print("--- FINAL TTS (NO CLAMP, PEAK PRESERVING) ---");

  final tokenizer = EnglishCharacterTokenizer();

  const int embedSize = 32;
  const int audioBins = 513;
  const int maxTextLen = 30;
  const int maxAudioLen = 64;
  const int sampleRate = 16000;

  final spectrogram = await stftSpectrogram(
    "output.wav",
    frameSize: 1024,
    hopSize: 256,
  );

  final target = spectrogram
      .take(maxAudioLen)
      .map((frame) => ValueVector.fromDoubleList(
            frame.map((v) => math.log(v + 1e-6)).toList(),
          ))
      .toList();

  // ✅ DEBUG TARGET
  print("\n--- TARGET SAMPLE ---");
  for (int i = 0; i < 10; i++) {
    print("target[$i] = ${target[0][i].data}");
  }

  final text = "IF HE'D RUN OUT OF";
  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  final model = TransformerDecoder(
    vocabSize: math.max(audioBins, maxAudioLen),
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final optimizer = SGD(model.parameters(), 0.003);
  final timeIdx = List.generate(maxAudioLen, (i) => i);

  // ✅ TRAIN
  for (int epoch = 1; epoch <= 600; epoch++) {
    optimizer.zeroGrad();

    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(embedSize, (i) => (t + i) / 10.0), // ✅ stronger signal
      );
    }).toList();

    final pred = model.forward(timeIdx, context);

    final losses = <Value>[];
    final len = math.min(pred.length, target.length);

    for (int i = 0; i < len; i++) {
      final diff = pred[i] - target[i];

      for (int k = 0; k < diff.length; k++) {
        // ✅ frequency-weighted MAE
        final weight = 1.0 + (k / audioBins);
        losses.add(diff[k].abs() * Value(weight));
      }
    }

    final loss = ValueVector(losses).sum() / Value(len * audioBins.toDouble());

    loss.backward();

    // ✅ minimal clipping (keep stable but not aggressive)
    // for (var p in model.parameters()) {
    //   p.grad = p.grad.clamp(-0.25, 0.25);
    // }

    optimizer.step();

    if (epoch % 20 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${loss.data}");
    }
  }

  print("Training done");

  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(embedSize, (i) => (t + i) / 10.0),
    );
  }).toList();

  final output = model.forward(timeIdx, context);

  // ✅ DEBUG PREDICTION
  print("\n--- PREDICTED ---");
  for (int i = 0; i < 10; i++) {
    print("pred[$i] = ${output[0][i].data}");
  }

  // ✅ NO CLAMP (as requested)
  final magnitudes = output.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) {
        final x = v.data;
        return math.exp(x * 2.0); // ✅ restore spectral contrast
      }).toList(),
    );
  }).toList();

  // ✅ DEBUG MAGNITUDES
  print("\n--- MAGNITUDES ---");
  for (int i = 0; i < 10; i++) {
    print("mag[$i] = ${magnitudes[0][i]}");
  }

  final griffin = GriffinLimGenerator(
    iterations: 80,
    frameSize: 1024,
    hopSize: 256,
  );

  final pcm = griffin.generateWav(magnitudes, sampleRate);

  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final file = File("tts_final_output.wav");
  encoder.encode(file, pcm);

  print("\n✅ Saved: ${file.absolute.path}");
}
