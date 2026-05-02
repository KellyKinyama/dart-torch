import 'dart:math' as math;

import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../transformer/transformer_decoder.dart';
import '../transformer_misc/multi_modal_generator.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'libri_speech_dataset.dart';
import 'multi_modal_buffer.dart';
import 'tokenizer.dart';
import 'trainer2.dart';

void main() async {
  final tokenizer = EnglishCharacterTokenizer();
  final dataset = LibriSpeechDataset(
      "C:/Users/kkinyama/Downloads/train-clean-100/", tokenizer);

  // CPU-Optimized Hyperparameters
  const int maxTextLen = 100;
  const int maxAudioLen = 200; // Increased for diverse samples
  const double initialLr = 0.001; // Lower for full dataset stability
  const int vocabSize = 50;
  const int audioMels = 32;
  const int commonEmbedSize = 32;

  // Initialize Generator (reuse your existing setup)
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
  final optimizer = SGD(generator.parameters(), initialLr);
  final videoFrames = List.generate(
      20, (i) => List.generate(64, (j) => math.Random().nextDouble()));
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  for (int epoch = 1; epoch <= 10; epoch++) {
    double epochLoss = 0;
    int count = 0;

    for (var sample in dataset.stream(maxTextLen, maxAudioLen)) {
      optimizer.zeroGrad();

      // 1. Load and Convert (if needed)
      // Use your existing 'save' and 'melSpectrogram' logic
      final rawSpectrogram = await melSpectrogram(sample['path'], nMels: 32);
      final audioInput =
          MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: maxAudioLen);

      // 2. Tokenize Text
      final targetTokens = tokenizer.encode(sample['text'], maxLen: maxTextLen);
      final inputTokens = targetTokens.sublist(0, targetTokens.length - 1);
      final expectedTargets = targetTokens.sublist(1);

      // 3. Forward & Backward
      final logits = generator.forward(audioInput, videoInput, inputTokens);
      Value totalLoss = Value(0.0);

      for (int i = 0; i < logits.length; i++) {
        final targetVector = ValueVector.fromDoubleList(List.generate(
            tokenizer.vocabSize,
            (idx) => idx == expectedTargets[i] ? 1.0 : 0.0));
        totalLoss += logits[i].softmax().crossEntropy(targetVector);
      }

      final normalizedLoss = totalLoss / Value(logits.length.toDouble());
      normalizedLoss.backward();

      // 4. Update
      optimizer.step();

      epochLoss += normalizedLoss.data;
      count++;

      if (count % 50 == 0) {
        print("Batch $count | Avg Loss: ${epochLoss / count}");
      }
    }
    print(
        "--- Epoch $epoch Completed | Final Avg Loss: ${epochLoss / count} ---");
  }
}
