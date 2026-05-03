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
  print("--- DEBUG TTS (GRIFFIN–LIM) ---");

  final tokenizer = EnglishCharacterTokenizer();

  const int embedSize = 32;
  const int audioBins = 513;
  const int maxTextLen = 30;
  const int maxAudioLen = 64;
  const int sampleRate = 16000;

  // ✅ LOAD STFT
  final spectrogram = await stftSpectrogram(
    "output.wav",
    frameSize: 1024,
    hopSize: 256,
  );

  // ✅ LOG TARGET
  final target = spectrogram
      .take(maxAudioLen)
      .map((frame) => ValueVector.fromDoubleList(
            frame.map((v) => math.log(v + 1e-6)).toList(),
          ))
      .toList();

  // ✅ DEBUG TARGET
  print("\n--- TARGET SAMPLE (frame 0) ---");
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

  final optimizer = SGD(model.parameters(), 0.005);
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
    final len = math.min(pred.length, target.length);

    for (int i = 0; i < len; i++) {
      final diff = pred[i] - target[i];
      losses.addAll(diff.squared().values);
    }

    final loss = ValueVector(losses).sum() / Value(len * audioBins.toDouble());

    loss.backward();

    // for (var p in model.parameters()) {
    //   p.grad = p.grad.clamp(-0.5, 0.5);
    // }

    optimizer.step();

    if (epoch % 5 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${loss.data}");
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

  // ✅ DEBUG PREDICTION
  print("\n--- PREDICTED (LOG SPACE, frame 0) ---");
  for (int i = 0; i < 10; i++) {
    print("pred[$i] = ${output[0][i].data}");
  }

  // ✅ LOG → MAG (NO NEW CLAMP)
  final magnitudes = output.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) {
        final x = v.data;
        return math.exp(x); // critical step
      }).toList(),
    );
  }).toList();

  // ✅ DEBUG MAGNITUDE
  print("\n--- MAGNITUDE (frame 0) ---");
  for (int i = 0; i < 10; i++) {
    print("mag[$i] = ${magnitudes[0][i]}");
  }

  // ✅ GRIFFIN–LIM
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

  final file = File("tts_debug_output.wav");
  encoder.encode(file, pcm);

  print("\n✅ Saved: ${file.absolute.path}");
}
