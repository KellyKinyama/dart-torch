import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer/transformer_decoder.dart';
import '../stt/tokenizer.dart';
import '../stt/audio_to_spectogram/audio_spectrogram.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

/// ------------------------------------------------------------
/// ✅ SIMPLE SGD
/// ------------------------------------------------------------
class SGD {
  final List<Value> parameters;
  final double lr;

  SGD(this.parameters, this.lr);

  void step() {
    for (final p in parameters) {
      p.data -= lr * p.grad;
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

/// ------------------------------------------------------------
/// ✅ INTELLIGIBLE VOCODER (voiced + noise)
/// ------------------------------------------------------------
class IntelligibleVocoder {
  final int frameSize;
  final int hopSize;
  final int sampleRate;

  IntelligibleVocoder({
    this.frameSize = 1024,
    this.hopSize = 256,
    this.sampleRate = 16000,
  });

  Uint8List generate(List<Float64List> mags) {
    final rand = math.Random();

    final signal = Float64List(mags.length * hopSize + frameSize);

    final window = _hann(frameSize);
    final bins = mags[0].length;

    final phases = List.filled(bins, 0.0);

    for (int f = 0; f < mags.length; f++) {
      final start = f * hopSize;

      // ✅ decide voiced/unvoiced
      double energy = mags[f].reduce((a, b) => a + b) / bins;
      bool voiced = energy > 0.02;

      for (int n = 0; n < frameSize; n++) {
        double sample = 0.0;

        for (int k = 1; k < bins; k++) {
          final mag = mags[f][k];
          final freq = k * sampleRate / frameSize;

          if (voiced) {
            phases[k] += 2 * math.pi * freq / sampleRate;
            sample += mag * 0.2 * math.sin(phases[k]);
          } else {
            sample += mag * 0.2 * (rand.nextDouble() * 2 - 1);
          }
        }

        signal[start + n] += sample * window[n];
      }
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

  Float64List _hann(int size) {
    return Float64List.fromList(List.generate(
      size,
      (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))),
    ));
  }
}

/// ------------------------------------------------------------
/// ✅ MAIN TTS
/// ------------------------------------------------------------
void main() async {
  print("------ SIMPLE TTS ------");

  final tokenizer = EnglishCharacterTokenizer();

  const embedSize = 32;
  const audioMels = 32;
  const maxTextLen = 100;
  const maxAudioLen = 120;

  const sampleRate = 16000;

  /// 1. PREP AUDIO TARGET
  final wavPath = "output.wav"; // must exist
  final spec = await melSpectrogram(
    wavPath,
    sampleRate: sampleRate,
    nMels: audioMels,
  );

  final target = spec
      .take(maxAudioLen)
      .map((f) => ValueVector.fromDoubleList(
            f.map((v) => v / 80.0).toList(), // ✅ normalize only
          ))
      .toList();

  /// 2. TEXT INPUT
  final text = "IF HE'D RUN OUT OF TURNIP SEED HE WOULDN'T DRESS UP";
  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  /// 3. MODEL
  final model = TransformerDecoder(
    vocabSize: audioMels,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final opt = SGD(model.parameters(), 0.01);
  final timeIdx = List.generate(maxAudioLen, (i) => i);

  /// 4. TRAIN
  for (int epoch = 1; epoch <= 200; epoch++) {
    opt.zeroGrad();

    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(embedSize, (i) => (t + i) / 100.0),
      );
    }).toList();

    final pred = model.forward(timeIdx, context);

    final losses = <Value>[];

    for (int i = 0; i < target.length; i++) {
      final diff = pred[i] - target[i];
      losses.addAll(diff.squared().values);
    }

    final loss = ValueVector(losses).sum();

    loss.backward();

    for (var p in model.parameters()) {
      p.grad = p.grad.clamp(-1.0, 1.0);
    }

    opt.step();

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${loss.data}");
    }
  }

  print("Training complete");

  /// 5. INFERENCE
  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(embedSize, (i) => (t + i) / 100.0),
    );
  }).toList();

  final output = model.forward(timeIdx, context);

  final mags = output.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) {
        final double x = (v.data as num).clamp(-1.0, 1.0).toDouble();
        final double db = x * 80.0; // reverse normalization
        return math.pow(10.0, db / 20.0).toDouble();
      }).toList(),
    );
  }).toList();

// double x = v.data.clamp(-1.0, 1.0);
// double db = x * 80.0;   // reverse normalization
// return math.pow(10.0, db / 20.0); // convert dB → magnitude

  /// 6. VOCODER
  final vocoder = IntelligibleVocoder(
    frameSize: 1024,
    hopSize: 256,
    sampleRate: sampleRate,
  );

  final pcm = vocoder.generate(mags);

  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final file = File("tts_output.wav");
  encoder.encode(file, pcm);

  print("Saved: ${file.absolute.path}");
}
