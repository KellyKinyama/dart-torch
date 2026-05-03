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
    int bins = mags[0].length;

    final phases = List.filled(bins, 0.0);

    for (int f = 0; f < mags.length; f++) {
      int start = f * hopSize;

      double energy = mags[f].reduce((a, b) => a + b) / bins;
      bool voiced = energy > 0.02;

      for (int n = 0; n < frameSize; n++) {
        double sample = 0.0;

        for (int k = 1; k < bins; k++) {
          double mag = mags[f][k];
          double freq = k * sampleRate / frameSize;

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

/// ---------------- MAIN ----------------
void main() async {
  print("------ AUTOREGRESSIVE TTS ------");

  final tokenizer = EnglishCharacterTokenizer();

  const embedSize = 32;
  const audioMels = 32;
  const maxTextLen = 100;
  const maxAudioLen = 80;

  const sampleRate = 16000;

  /// ✅ LOAD TARGET AUDIO
  final wavPath = "output.wav";

  final spec = await melSpectrogram(
    wavPath,
    sampleRate: sampleRate,
    nMels: audioMels,
  );

  final target = spec
      .take(maxAudioLen)
      .map((f) => ValueVector.fromDoubleList(
            f.map((v) {
              // v is dB in [-80, 0]
              double x = (v + 80.0) / 80.0; // [0,1]
              return x * 2.0 - 1.0; // ✅ [-1,1]
            }).toList(),
          ))
      .toList();

  /// ✅ TEXT INPUT
  final text = "IF HE'D RUN OUT OF TURNIP SEED HE WOULDN'T DRESS UP";

  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  /// ✅ MODEL (FIXED CONFIG)

  final model = TransformerDecoder(
    vocabSize: math.max(audioMels, maxAudioLen), // ✅ FIX
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final opt = SGD(model.parameters(), 0.005);

  /// ✅ TRAIN LOOP
  for (int epoch = 1; epoch <= 150; epoch++) {
    opt.zeroGrad();

    // ✅ TEXT CONTEXT
    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(embedSize, (i) => (t + i) / 100.0),
      );
    }).toList();

    Value loss = Value(0.0);

    // ✅ AUTOREGRESSIVE LOOP
    List<int> prevSteps = [0];

    for (int t = 0; t < target.length - 1; t++) {
      final pred = model.forward(prevSteps, context);

      final outVec = pred.last;

      final targetVec = ValueVector.fromDoubleList(
        target[t].values.map((v) => math.log(v.data + 1e-6)).toList(),
      );

      final diff = outVec - targetVec;
      loss += diff.squared().mean();

      // ✅ TEACHER FORCING
      prevSteps.add(t % audioMels);
    }

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

  /// ✅ INFERENCE (AUTOREGRESSIVE)
  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(embedSize, (i) => (t + i) / 100.0),
    );
  }).toList();

  List<int> generatedSteps = [0];
  List<Float64List> mags = [];

  for (int t = 0; t < maxAudioLen; t++) {
    final pred = model.forward(generatedSteps, context);

    final vec = pred.last;

    final mag = Float64List.fromList(
      vec.values.map((v) {
        double x = v.data.clamp(-10.0, 5.0);
        return math.exp(x) * 3.0;
      }).toList(),
    );

    mags.add(mag);

    // ✅ next step token
    generatedSteps.add(t % audioMels);
  }

  /// ✅ VOCODER
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

  final file = File("tts_autoreg.wav");
  encoder.encode(file, pcm);

  print("Saved: ${file.absolute.path}");
}
