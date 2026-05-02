import 'dart:math' as math;
import 'dart:typed_data';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer/example_aft_full_cycle3.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../transformer_misc/multi_modal_generator.dart';
import '../transformer/transformer_decoder.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';

import 'dart:io';
import 'dart:typed_data';

import 'package:audio_codec/src/flac/flac_decoder.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

void save(String flacPath) {
  final flacFile = File(flacPath);

  final decoder = FlacDecoder(track: flacFile);
  final result = decoder.decode();

  final pcmSamples = Int32List(
    result.streamInfoBlock!.totalSamples * result.streamInfoBlock!.channels,
  );

  int frameNumber = 0;

  while (decoder.hasNextFrame()) {
    final frame = decoder.readFrame();

    writeFrameToPcm(
      pcmSamples,
      frame,
      frameNumber,
      result.streamInfoBlock!.sampleRate,
    );

    frameNumber++;
  }

  decoder.close();

  // WavEncoder.encode expects bytes; convert 32-bit PCM ints to little-endian bytes.
  final pcmBytes = Uint8List.view(pcmSamples.buffer);

  WavEncoder(
    sampleRate: result.streamInfoBlock!.sampleRate,
    numChannels: result.streamInfoBlock!.channels,
    bitDepth: result.streamInfoBlock!.bitsPerSample,
  ).encode(
    File("output.wav"),
    pcmBytes,
  );
}

void main(List<String> args) async {
  print("--- Overfitting MultimodalGenerator with Stable Tensor-Engine ---");

  // Configuration
  const int commonEmbedSize = 32;
  const int vocabSize = 50;
  const int maxTextLen = 12;
  const int audioMels = 32;
  // Change this line in trainer.dart

  // Use the absolute path confirmed in your PS output
  const String corpusRoot =
      "C:/Users/kkinyama/Downloads/train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac"; // final String flacPath =
  //     "C:/Users/kkinyama/Downloads/train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac";

// Select one specific utterance for your 5-sample overfit test
// Example: 103-1240-0015.flac (The shorter 117-frame sample is great for fast overfitting)
  String wavPath = "output.wav";

  save(corpusRoot);

  wavPath = wavPath.replaceAll(".flac", ".wav");
  // Model Setup
  final audioModel = AudioTransformer(
    featureDim: audioMels,
    embedSize: commonEmbedSize,
    maxAudioSequenceLength: 64,
    numClasses: 1,
    numLayers: 1,
    numHeads: 2,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: 64,
    embedSize: commonEmbedSize,
    maxVideoSequenceLength: 20,
    numClasses: 1,
    numLayers: 1,
    numHeads: 2,
  );

  final decoder = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: commonEmbedSize,
    encoderEmbedSize: commonEmbedSize,
    blockSize: maxTextLen,
    numLayers: 1,
    numHeads: 2,
  );

  final generator = MultimodalGenerator(
    audioEncoder: audioModel,
    videoEncoder: videoModel,
    decoder: decoder,
  );

  // Data Preparation from your earlier loader success
  // Assuming melSpectrogram helper is available
  final rawSpectrogram =
      await melSpectrogram(wavPath, sampleRate: 16000, nMels: audioMels);
  final audioInput = MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: 64);

  final videoFrames = List.generate(
      20, (i) => List.generate(64, (j) => math.Random().nextDouble()));
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  // Targets: <SOS>, THE, TRANSFORMER, NOMINAL, <EOS>
  final List<int> targetTokens = [0, 5, 12, 3, 1];
  final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
  final expectedTargets = targetTokens.sublist(1);

  // Optimization Setup
  final optimizer = SGD(generator.parameters(),
      0.01); // Slightly higher LR for faster overfitting

  for (int epoch = 1; epoch <= 200; epoch++) {
    optimizer.zeroGrad();

    // 1. Forward Pass
    final List<ValueVector> logits =
        generator.forward(audioInput, videoInput, inputTokens);

    Value totalLoss = Value(0.0);
    for (int i = 0; i < logits.length; i++) {
      final targetVector = ValueVector.fromDoubleList(List.generate(
          vocabSize, (idx) => idx == expectedTargets[i] ? 1.0 : 0.0));

      final probs = logits[i].softmax();
      totalLoss += probs.crossEntropy(targetVector);
    }

    final normalizedLoss = totalLoss / Value(logits.length.toDouble());

    // 2. Backward Pass
    normalizedLoss.backward();

    // 3. Gradient Clipping: Prevent NaNs in pure Dart math
    for (var p in generator.parameters()) {
      p.grad = p.grad.clamp(-1.0, 1.0);
    }

    // 4. Parameter Update
    optimizer.step();

    if (epoch % 20 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${normalizedLoss.data.toStringAsFixed(6)}");
    }
  }

  print("/n--- Overfitting Complete ---");
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

void writeFrameToPcm(
    Int32List pcmSamples, FlacFrame frame, int frameNumber, int sampleRate) {
  // Use the count property from the AudioChannelLayout
  final int channels = frame.channels.nbChannels;
  final int blockSize = frame.blockSize;
  final int offset = frameNumber * blockSize * channels;

  for (int i = 0; i < blockSize; i++) {
    for (int c = 0; c < channels; c++) {
      // Access the subframe samples for the specific channel.
      // `Samples` is indexable (no `.data` getter).
      pcmSamples[offset + (i * channels) + c] = frame.subframes[c][i];
    }
  }
}
