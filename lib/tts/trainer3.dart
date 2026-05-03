import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../stft_spectrogram.dart';
import '../stt/tokenizer.dart';
import '../transformer/transformer_decoder.dart';
import 'simple_vocoder.dart'; // ✅ NEW
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
  print("--- FIXED LEAN TTS (VOCODER) ---");

  final tokenizer = EnglishCharacterTokenizer();

  const int embedSize = 32;
  const int audioBins = 513;
  const int maxTextLen = 30;
  const int maxAudioLen = 32;
  const int sampleRate = 16000;

  // ✅ LOAD STFT
  final spectrogram = await stftSpectrogram("output.wav");

  final target = spectrogram
      .take(maxAudioLen)
      .map((frame) => ValueVector.fromDoubleList(
            frame.map((v) => math.log(v + 1e-6)).toList(),
          ))
      .toList();

  String text = "IF HE'D RUN OUT OF";
  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  final model = TransformerDecoder(
    vocabSize: audioBins,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final optimizer = SGD(model.parameters(), 0.01);
  final timeIdx = List.generate(maxAudioLen, (i) => i);

  // ✅ TRAIN
  for (int epoch = 1; epoch <= 200; epoch++) {
    optimizer.zeroGrad();

    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(
          embedSize,
          (i) => (t + i) / (tokenizer.vocabSize + embedSize),
        ),
      );
    }).toList();

    final pred = model.forward(timeIdx, context);

    final losses = <Value>[];
    int len = math.min(pred.length, target.length);

    for (int i = 0; i < len; i++) {
      final diff = pred[i] - target[i];
      losses.addAll(diff.squared().values);
    }

    final totalLoss = ValueVector(losses).sum();
    final normalizedLoss = totalLoss / Value(len * audioBins.toDouble());

    normalizedLoss.backward();

    for (var p in model.parameters()) {
      p.grad = p.grad.clamp(-0.5, 0.5);
    }

    optimizer.step();

    if (epoch % 5 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${normalizedLoss.data}");
    }
  }

  print("Training done");

  // ✅ INFERENCE
  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(
        embedSize,
        (i) => (t + i) / (tokenizer.vocabSize + embedSize),
      ),
    );
  }).toList();

  final output = model.forward(timeIdx, context);

  // ✅ LOG → MAG
  final magnitudes = output.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) {
        double x = v.data.clamp(-10.0, 5.0);
        return math.exp(x);
      }).toList(),
    );
  }).toList();

  // ✅ ✅ NEW VOCODER (REPLACES GRIFFIN-LIM)
  final vocoder = SimpleVocoder(
    frameSize: 1024,
    hopSize: 256,
  );

  final pcm = vocoder.generate(magnitudes, sampleRate);

  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final file = File("vocoder_output.wav");
  encoder.encode(file, pcm);

  print("Saved: ${file.absolute.path}");
}
