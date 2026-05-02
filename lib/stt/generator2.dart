import 'dart:math' as math;
import '../nn/value.dart';
import '../nn/value_vector.dart';
import '../transformer_misc/audio_transformer.dart';
import '../transformer_misc/aft_video_transformer.dart';
import '../transformer_misc/multi_modal_generator.dart';
import '../transformer/transformer_decoder.dart';
import 'audio_to_spectogram/audio_spectrogram.dart';
import 'multi_modal_buffer.dart';

// --- Tokenizer Implementation ---
class EnglishCharacterTokenizer {
  final Map<String, int> _charToId = {};
  final Map<int, String> _idToChar = {};
  static const String pad = '<PAD>', sos = '<SOS>', eos = '<EOS>', unk = '<UNK>';

  EnglishCharacterTokenizer() {
    [pad, sos, eos, unk].forEach(_addToken);
    _addRange(97, 122); // a-z
    _addRange(65, 90);  // A-Z
    _addRange(48, 57);  // 0-9
    " !\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~".split('').forEach(_addToken);
  }

  void _addToken(String char) {
    if (!_charToId.containsKey(char)) {
      final id = _charToId.length;
      _charToId[char] = id;
      _idToChar[id] = char;
    }
  }

  void _addRange(int start, int end) => 
      Iterable.generate(end - start + 1, (i) => String.fromCharCode(start + i)).forEach(_addToken);

  int get vocabSize => _charToId.length;
  int get sosId => _charToId[sos]!;
  int get eosId => _charToId[eos]!;

  List<int> encode(String text, {int? maxLen}) {
    List<int> ids = [_charToId[sos]!, ...text.split('').map((c) => _charToId[c] ?? _charToId[unk]!), _charToId[eos]!];
    if (maxLen != null) {
      if (ids.length > maxLen) ids = ids.sublist(0, maxLen - 1)..add(_charToId[eos]!);
      while (ids.length < maxLen) ids.add(_charToId[pad]!);
    }
    return ids;
  }

  String decode(List<int> ids) => ids
      .map((id) => _idToChar[id] ?? unk)
      .where((char) => ![pad, sos, eos].contains(char))
      .join('');
}

// Simple SGD Implementation
class SGD {
  final List<Value> parameters;
  final double learningRate;
  SGD(this.parameters, this.learningRate);
  void step() => parameters.forEach((p) => p.data -= learningRate * p.grad);
  void zeroGrad() => parameters.forEach((p) => p.grad = 0.0);
}

void main(List<String> args) async {
  print("--- Multimodal Training with English Character Tokenizer ---");

  // --- 1. Tokenizer & Config ---
  final tokenizer = EnglishCharacterTokenizer();
  final int vocabSize = tokenizer.vocabSize;
  const int commonEmbedSize = 32;
  const int maxTextLen = 20; // Increased to fit longer strings
  const int audioMels = 32;
  String wavPath = args.isNotEmpty ? args[0] : "test.wav";

  // --- 2. Model Instantiation ---
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

  // --- 3. Data Preparation ---
  final rawSpectrogram = await melSpectrogram(wavPath, sampleRate: 16000, nMels: audioMels);
  final audioInput = MultimodalBuffer.prepareAudio(rawSpectrogram, maxLen: 64);
  final List<List<double>> videoFrames = List.generate(20, (i) => List.generate(64, (j) => math.Random().nextDouble()));
  final videoInput = MultimodalBuffer.prepareVideo(videoFrames, maxLen: 20);

  // NEW: Encode a string using the tokenizer
  final String targetSentence = "ZESCO-AI"; 
  final List<int> tokens = tokenizer.encode(targetSentence, maxLen: maxTextLen);
  
  // Shift for Teacher Forcing
  final inputTokens = tokens.sublist(0, tokens.length - 1);
  final expectedTargets = tokens.sublist(1);

  // --- 4. Training Loop ---
  final optimizer = SGD(generator.parameters(), 0.05); // Faster for overfitting

  for (int epoch = 1; epoch <= 100; epoch++) {
    optimizer.zeroGrad();

    final List<ValueVector> logits = generator.forward(audioInput, videoInput, inputTokens);

    Value totalLoss = Value(0.0);
    for (int i = 0; i < logits.length; i++) {
      final targetVector = ValueVector.fromDoubleList(
          List.generate(vocabSize, (idx) => idx == expectedTargets[i] ? 1.0 : 0.0));
      totalLoss += logits[i].softmax().crossEntropy(targetVector);
    }

    final normalizedLoss = totalLoss / Value(logits.length.toDouble());
    normalizedLoss.backward();
    optimizer.step();

    if (epoch % 10 == 0 || epoch == 1) {
      print("Epoch $epoch | Loss: ${normalizedLoss.data.toStringAsFixed(6)}");
    }
  }

  print("\n--- Overfitting Complete ---");
  verifyInference(generator, audioInput, videoInput, tokenizer, maxTextLen);
}

void verifyInference(MultimodalGenerator gen, List<ValueVector> audio, List<ValueVector> video, EnglishCharacterTokenizer tokenizer, int maxLen) {
  List<int> current = [tokenizer.sosId];
  
  for (int i = 0; i < maxLen; i++) {
    final logits = gen.forward(audio, video, current);
    final nextId = logits.last
        .softmax()
        .values
        .asMap()
        .entries
        .reduce((a, b) => a.value.data > b.value.data ? a : b)
        .key;
    
    current.add(nextId);
    if (nextId == tokenizer.eosId) break;
  }

  print("Target Sentence: ZESCO-AI");
  print("Predicted IDs: $current");
  print("Decoded Output: ${tokenizer.decode(current)}");
}