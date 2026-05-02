import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../transformer_misc/multi_modal_generator.dart';
import '../transformer/transformer_decoder.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';
// Ensure your tokenizer path is correct
import 'tokenizer.dart';

import 'package:audio_codec/src/flac/flac_decoder.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

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
        pcmSamples, frame, frameNumber, result.streamInfoBlock!.sampleRate);
    frameNumber++;
  }
  decoder.close();

  final pcmBytes = Uint8List.view(pcmSamples.buffer);
  WavEncoder(
    sampleRate: result.streamInfoBlock!.sampleRate,
    numChannels: result.streamInfoBlock!.channels,
    bitDepth: result.streamInfoBlock!.bitsPerSample,
  ).encode(File("output.wav"), pcmBytes);
}

void writeFrameToPcm(
    Int32List pcmSamples, FlacFrame frame, int frameNumber, int sampleRate) {
  final int channels = frame.channels.nbChannels;
  final int blockSize = frame.blockSize;
  final int offset = frameNumber * blockSize * channels;

  for (int i = 0; i < blockSize; i++) {
    for (int c = 0; c < channels; c++) {
      // Accessing subframes samples directly as indexable
      pcmSamples[offset + (i * channels) + c] = frame.subframes[c][i];
    }
  }
}

void main(List<String> args) async {
  print("--- Overfitting MultimodalGenerator: Full Vocabulary Mode ---");

  // 1. Initialize Tokenizer to get actual Vocabulary Size
  final tokenizer = EnglishCharacterTokenizer();
  final int fullVocabSize = tokenizer.vocabSize;
  print("Detected Vocabulary Size: $fullVocabSize");

  // const int commonEmbedSize = 32;
  // const int maxTextLen = 20; // Increased for actual sentences
  const int audioMels = 32;

  const int commonEmbedSize = 32;
  const int maxTextLen = 120; // Increased to fit the full sentence (93 chars)
  const int maxAudioLen = 150; // Ensure this covers all 117 frames of the audio
// const int audioMels = 32;

  const String corpusRoot =
      "C:/Users/kkinyama/Downloads/train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac";
  save(corpusRoot);
  String wavPath = "output.wav";

  // 2. Model Setup with Full Vocab
  final audioModel = AudioTransformer(
    featureDim: audioMels,
    embedSize: commonEmbedSize,
    // maxAudioSequenceLength: 128, // Increased for longer utterances
    maxAudioSequenceLength: maxAudioLen, // Matc
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
    vocabSize: fullVocabSize, // Expanded!
    embedSize: commonEmbedSize,
    encoderEmbedSize: commonEmbedSize,
    // blockSize: maxTextLen,
    blockSize: maxAudioLen,
    numLayers: 1,
    numHeads: 2,
  );

  final generator = MultimodalGenerator(
    audioEncoder: audioModel,
    videoEncoder: videoModel,
    decoder: decoder,
  );

  // 3. Data Prep
  final rawSpectrogram =
      await melSpectrogram(wavPath, sampleRate: 16000, nMels: audioMels);
  final audioInput = MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: 128);

  final videoFrames = List.generate(
      20, (i) => List.generate(64, (j) => math.Random().nextDouble()));
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  // 4. Tokenize actual LibriSpeech Label
  // String labelText = "IF HE'D RUN OUT OF TURNIP SEED";
  // final List<int> targetTokens =
  //     tokenizer.encode(labelText, maxLen: maxTextLen);

  // final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
  // final expectedTargets = targetTokens.sublist(1);

  // The full transcript for 103-1240-0015
  String labelText =
      "IF HE'D RUN OUT OF TURNIP SEED HE WOULDN'T DRESS UP AND TAKE THE BUGGY TO GO FOR MORE";

// Tokenize the full string
  final List<int> targetTokens =
      tokenizer.encode(labelText, maxLen: maxTextLen);

  final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
  final expectedTargets = targetTokens.sublist(1);

  final optimizer = SGD(generator.parameters(), 0.01);

  for (int epoch = 1; epoch <= 150; epoch++) {
    optimizer.zeroGrad();

    final List<ValueVector> logits =
        generator.forward(audioInput, videoInput, inputTokens);

    Value totalLoss = Value(0.0);
    for (int i = 0; i < logits.length; i++) {
      // Use the expanded fullVocabSize for the target vector
      final targetVector = ValueVector.fromDoubleList(List.generate(
          fullVocabSize, (idx) => idx == expectedTargets[i] ? 1.0 : 0.0));

      final probs = logits[i].softmax();
      totalLoss += probs.crossEntropy(targetVector);
    }

    final normalizedLoss = totalLoss / Value(logits.length.toDouble());
    normalizedLoss.backward();

    for (var p in generator.parameters()) {
      p.grad = p.grad.clamp(-1.0, 1.0);
    }

    optimizer.step();

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${normalizedLoss.data.toStringAsFixed(6)}");
    }
  }

  print("\n--- Overfitting Complete ---");
  verifyInference(generator, audioInput, videoInput, tokenizer);
}

void verifyInference(MultimodalGenerator gen, List<ValueVector> audio,
    List<ValueVector> video, EnglishCharacterTokenizer tokenizer) {
  List<int> current = [tokenizer.sosTokenId];
  for (int i = 0; i < 20; i++) {
    final logits = gen.forward(audio, video, current);
    final nextId = logits.last
        .softmax()
        .values
        .asMap()
        .entries
        .reduce((a, b) => a.value.data > b.value.data ? a : b)
        .key;
    current.add(nextId);
    if (nextId == tokenizer.eosTokenId) break;
  }
  print("Predicted Tokens: $current");
  print("Decoded String: ${tokenizer.decode(current)}");
}
