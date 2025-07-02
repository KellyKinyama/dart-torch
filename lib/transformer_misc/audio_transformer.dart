// file: audio_transformer.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer2/transformer_encoder2.dart'; // Your TransformerEncoder

/// A Transformer model adapted for audio classification.
///
/// This model expects pre-extracted audio features (e.g., MFCCs,
/// spectrogram slices) as a sequence of ValueVectors.
class AudioTransformer extends Module {
  final int
      featureDim; // Dimension of each audio feature vector (e.g., 40 for MFCCs)
  final int embedSize; // Transformer embedding dimension
  final int
      maxAudioSequenceLength; // Max number of feature vectors in a sequence
  final int numClasses; // Number of output classes for audio classification
  final int numLayers; // Number of encoder layers
  final int numHeads; // Number of attention heads

  // Linear projection from audio feature dimension to Transformer embedSize
  final Layer featureProjection;

  // Positional embeddings (learned)
  final List<ValueVector> positionEmbeddings;

  // The main Transformer Encoder backbone
  final TransformerEncoder transformerEncoder;

  // Final MLP head for classification
  final Layer mlpHead;

  AudioTransformer({
    required this.featureDim,
    required this.embedSize,
    required this.maxAudioSequenceLength,
    required this.numClasses,
    this.numLayers = 2, // Reduced for faster example execution
    this.numHeads = 4, // Reduced for faster example execution
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        // Project audio features to embedSize
        featureProjection = Layer.fromNeurons(featureDim, embedSize),
        // Position embeddings for the sequence of audio feature vectors
        positionEmbeddings = List.generate(
            maxAudioSequenceLength,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // The TransformerEncoder processes the embedded audio features
        transformerEncoder = TransformerEncoder(
          vocabSize: 0, // Not used directly
          embedSize: embedSize,
          blockSize: maxAudioSequenceLength,
          numLayers: numLayers,
          numHeads: numHeads,
        ),
        // Classification head
        mlpHead = Layer.fromNeurons(embedSize, numClasses);

  /// The forward pass for the Audio Transformer.
  ///
  /// Takes a list of pre-extracted audio feature vectors.
  /// Returns logits for audio classification.
  List<Value> forward(List<ValueVector> audioFeatures) {
    if (audioFeatures.length > maxAudioSequenceLength) {
      throw ArgumentError(
          "Audio sequence length (${audioFeatures.length}) exceeds "
          "maxAudioSequenceLength ($maxAudioSequenceLength).");
    }

    // 1. Project raw audio features to Transformer embedSize
    final embeddedFeatures =
        audioFeatures.map((f) => featureProjection.forward(f)).toList();

    // 2. Add positional embeddings
    final sequenceWithPositionalEmbeddings =
        List.generate(embeddedFeatures.length, (i) {
      return embeddedFeatures[i] + positionEmbeddings[i];
    });

    // 3. Pass the sequence through the Transformer Encoder
    final encodedFeatures =
        transformerEncoder.forwardEmbeddings(sequenceWithPositionalEmbeddings);

    // 4. Global Average Pooling over the sequence to get a single vector for classification
    // Alternatively, you could use a [CLS] token like in ViT.
    // For simplicity, we'll average here.
    ValueVector pooledFeature;
    if (encodedFeatures.isEmpty) {
      pooledFeature = ValueVector(
          List.filled(embedSize, Value(0.0))); // Handle empty sequence
    } else {
      pooledFeature = encodedFeatures.reduce((a, b) => a + b) /
          Value(encodedFeatures.length.toDouble());
    }

    // 5. Pass through the MLP head for classification logits
    final logits = mlpHead.forward(pooledFeature);

    return logits
        .values; // Return a list of Value objects for a single prediction
  }

  @override
  List<Value> parameters() {
    return [
      ...featureProjection.parameters(),
      ...positionEmbeddings.expand((vec) => vec.values),
      ...transformerEncoder.parameters(),
      ...mlpHead.parameters(),
    ];
  }
}
