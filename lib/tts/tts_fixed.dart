import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer/transformer_decoder.dart';
import '../stt/tokenizer.dart';
import '../stt/audio_to_spectogram/audio_spectrogram.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

/// ---------------- SGD ----------------
class SGD {
  final List<Value> params;
  final double lr;
  SGD(this.params, this.lr);

  void step() {
    for (var p in params) {
      p.data -= lr * p.grad;
    }
  }

  void zeroGrad() {
    for (var p in params) {
      p.grad = 0.0;
    }
  }
}

/// ---------------- VOCODER ----------------
class SimpleVocoder {
  Uint8List generate(List<Float64List> mags) {
    final rand = math.Random();
    final signal = Float64List(mags.length * 256 + 1024);

    for (int i = 0; i < signal.length; i++) {
      signal[i] = rand.nextDouble() * 2 - 1;
    }

    return _toWav(signal);
  }

  Uint8List _toWav(Float64List signal) {
    double maxVal = 0.0;
    for (var s in signal) {
      if (s.abs() > maxVal) maxVal = s.abs();
    }

    final pcm = Int16List(signal.length);

    for (int i = 0; i < signal.length; i++) {
      pcm[i] = ((signal[i] / (maxVal + 1e-9)) * 32767).toInt();
    }

    return Uint8List.view(pcm.buffer);
  }
}

/// ---------------- MAIN ----------------
void main() async {
  print("------ FIXED TTS ------");

  final tokenizer = EnglishCharacterTokenizer();

  const embedSize = 32;
  const audioMels = 32;
  const maxTextLen = 100;
  const maxAudioLen = 80;
  const sampleRate = 16000;

  /// ✅ LOAD MEL (NO dB MISUSE)
  final spec = await melSpectrogram(
    "output.wav",
    sampleRate: sampleRate,
    nMels: audioMels,
  );

  /// ✅ CORRECT TARGET (log-mel)
  final target = spec.take(maxAudioLen).map((frame) {
    return ValueVector.fromDoubleList(
      frame.map((v) {
        double safe = math.max(v, 1e-6);
        double logMel = math.log(safe);         // ✅ ONE log only
        return logMel.clamp(-10.0, 2.0) / 10.0; // ✅ normalize stable
      }).toList(),
    );
  }).toList();

  /// ✅ TEXT INPUT
  final text = "IF HE'D RUN OUT OF TURNIP SEED";
  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  /// ✅ MODEL (FIXED SIZE)
  final model = TransformerDecoder(
    vocabSize: math.max(audioMels, maxAudioLen),
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final opt = SGD(model.parameters(), 0.001);
  final timeIdx = List.generate(maxAudioLen, (i) => i);

  /// ✅ TRAIN
  for (int epoch = 1; epoch <= 150; epoch++) {
    opt.zeroGrad();

    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(embedSize, (i) => (t + i) / 100.0),
      );
    }).toList();

    final pred = model.forward(timeIdx, context);

    /// ✅ CLAMP OUTPUT (critical)
    for (var vec in pred) {
      for (var v in vec.values) {
        if (v.data.isNaN || v.data.isInfinite) {
          v.data = 0.0;
        } else {
          v.data = v.data.clamp(-1.0, 1.0);
        }
      }
    }

    final losses = <Value>[];

    for (int i = 0; i < target.length; i++) {
      final diff = pred[i] - target[i];

      for (var v in diff.squared().values) {
        v.data = v.data.clamp(0.0, 10.0); // ✅ stability
        losses.add(v);
      }
    }

    final loss = ValueVector(losses).sum();

    if (loss.data.isNaN || loss.data.isInfinite) {
      print("NaN detected — skipping step");
      continue;
    }

    loss.backward();

    for (var p in model.parameters()) {
      p.grad = p.grad.clamp(-0.1, 0.1);
    }

    opt.step();

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${loss.data}");
    }
  }

  print("Training complete");

  /// ✅ INFERENCE
  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(embedSize, (i) => (t + i) / 100.0),
    );
  }).toList();

  final output = model.forward(timeIdx, context);

  final mags = output.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) {
        double x = v.data.clamp(-1.0, 1.0);
        double logMel = x * 10.0;
        return math.exp(logMel);
      }).toList(),
    );
  }).toList();

  final vocoder = SimpleVocoder();

  final pcm = vocoder.generate(mags);

  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final file = File("tts_fixed.wav");
  encoder.encode(file, pcm);

  print("Saved: ${file.absolute.path}");
}
