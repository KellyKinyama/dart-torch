import 'dart:io';
import 'dart:math' as math;
import '../face_rec/aft_vit_spectorgram_refiner.dart';
import 'dart:typed_data';
import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../stft_spectrogram.dart';
import '../stt/tokenizer.dart';
import '../transformer/transformer_decoder.dart';
import 'griffin_lim_generator.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

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

/// ✅ Balanced sum to avoid deep + chains (stack overflow) and reduce GC pressure.
Value _sumBalanced(List<Value> xs) {
  if (xs.isEmpty) return Value(0.0);
  if (xs.length == 1) return xs[0];

  List<Value> next = [];
  for (int i = 0; i < xs.length; i += 2) {
    if (i + 1 < xs.length) {
      next.add(xs[i] + xs[i + 1]);
    } else {
      next.add(xs[i]);
    }
  }
  return _sumBalanced(next);
}

/// ✅ abs(x) built from relu so we don't depend on Value.abs() existing
Value _absValue(Value x, Value negOne) {
  // abs(x) = relu(x) + relu(-x)
  return x.relu() + (x * negOne).relu();
}

void main() async {
  print("--- FINAL TTS: SMALL VIT PIPELINE (CPU/RAM FRIENDLY) ---");

  final tokenizer = EnglishCharacterTokenizer();

  // ---- Core sizes ----
  const int textEmbedSize = 32;
  const int audioBins = 513;
  const int maxAudioLen = 64; // you can lower to 32 for faster dev
  const int maxTextLen = 30;
  const int sampleRate = 16000;

  // ---- ✅ SMALLER ViT ----
  // Key: embedSize must be divisible by numHeads
  const int vitEmbedSize = 256; // was 512
  const int vitHeads = 4;
  const int vitLayers = 1; // was 2
  const int patchTime = 1; // was 4  (HUGE memory saver)

  // ---- ✅ Griffin cheaper ----
  const int glIterations = 30; // was 80 (you can raise later)

  // ✅ LOAD STFT magnitudes
  final spectrogram = await stftSpectrogram(
    "output.wav",
    frameSize: 1024,
    hopSize: 256,
  );

  // target in log-mag space
  final target = spectrogram
      .take(maxAudioLen)
      .map((f) => ValueVector.fromDoubleList(
            f.map((v) => math.log(v + 1e-6)).toList(),
          ))
      .toList();

  final text = "IF HE'D RUN OUT OF";
  final tokens = tokenizer.encode(text, maxLen: maxTextLen);

  // coarse transformer
  final model = TransformerDecoder(
    vocabSize: math.max(audioBins, maxAudioLen),
    embedSize: textEmbedSize,
    encoderEmbedSize: textEmbedSize,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  // ViT refiner (patches along time)
  final vit = ViTSpectrogramRefiner(
    timeSteps: maxAudioLen,
    freqBins: audioBins,
    patchTime: patchTime,
    embedSize: vitEmbedSize,
    numLayers: vitLayers,
    numHeads: vitHeads,
  );

  final optimizer = SGD(
    [
      ...model.parameters(),
      ...vit.parameters(),
    ],
    0.003,
  );

  final timeIdx = List.generate(maxAudioLen, (i) => i);

  // Precompute constant weights (no grads needed)
  final weightVals = List<Value>.generate(
    audioBins,
    (k) => Value(1.0 + (k / audioBins)),
  );

  final negOne = Value(-1.0);

  // ✅ TRAIN
  for (int epoch = 1; epoch <= 200; epoch++) {
    optimizer.zeroGrad();

    final context = tokens.map((t) {
      return ValueVector.fromDoubleList(
        List.generate(textEmbedSize, (i) => (t + i) / 10.0),
      );
    }).toList();

    // coarse spectrogram (log-mag)
    final coarse = model.forward(timeIdx, context);

    // flatten [T][F] -> [T*F]
    final flat = coarse.expand((f) => f.values.map((v) => v.data)).toList();

    // refine (ViT forward expects List<double> input)
    final refined = vit.forward(flat);

    final len = math.min(refined.length, target.length);

    // ✅ memory-friendly loss: compute per-frame sums, then balanced-reduce
    final frameLosses = <Value>[];

    for (int i = 0; i < len; i++) {
      final diff = refined[i] - target[i];

      final terms = <Value>[];
      for (int k = 0; k < diff.length; k++) {
        final absErr = _absValue(diff[k], negOne);
        terms.add(absErr * weightVals[k]);
      }

      frameLosses.add(_sumBalanced(terms));
    }

    final totalLoss = _sumBalanced(frameLosses);
    final loss = totalLoss / Value(len * audioBins.toDouble());

    loss.backward();

    // keep your grad clipping
    for (final p in optimizer.parameters) {
      p.grad = p.grad.clamp(-0.25, 0.25);
    }

    optimizer.step();

    if (epoch % 20 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${loss.data}");
    }
  }

  print("Training complete");

  // ✅ INFERENCE
  final context = tokens.map((t) {
    return ValueVector.fromDoubleList(
      List.generate(textEmbedSize, (i) => (t + i) / 10.0),
    );
  }).toList();

  final coarse = model.forward(timeIdx, context);
  final flat = coarse.expand((f) => f.values.map((v) => v.data)).toList();
  final refined = vit.forward(flat);

  // convert log-mag -> magnitude for Griffin-Lim
  final mags = refined.map((vec) {
    return Float64List.fromList(
      vec.values.map((v) => math.exp(v.data)).toList(),
    );
  }).toList();

  final griffin = GriffinLimGenerator(
    iterations: glIterations,
    frameSize: 1024,
    hopSize: 256,
  );

  final pcm = griffin.generateWav(mags, sampleRate);

  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  final file = File("tts_vit_small.wav");
  encoder.encode(file, pcm);

  print("✅ Saved: ${file.absolute.path}");
}
