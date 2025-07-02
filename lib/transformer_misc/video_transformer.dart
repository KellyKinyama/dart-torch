// file: video_transformer.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer2/transformer_encoder2.dart'; // Your TransformerEncoder

/// A Transformer model adapted for video classification (e.g., action recognition).
///
/// This model expects pre-extracted video frame or clip embeddings
/// as a sequence of ValueVectors. These embeddings could come from
/// a pre-trained CNN (like ResNet) or a Vision Transformer applied per frame.
class VideoTransformer extends Module {
  final int
      frameEmbedDim; // Dimension of each pre-extracted frame/clip embedding
  final int
      embedSize; // Transformer embedding dimension (often same as frameEmbedDim)
  final int
      maxVideoSequenceLength; // Max number of frames/clips in a video sequence
  final int numClasses; // Number of output classes for video classification
  final int numLayers; // Number of encoder layers
  final int numHeads; // Number of attention heads

  // Optional: Linear projection if frameEmbedDim != embedSize
  final Layer? frameProjection;

  // Positional embeddings (learned)
  final List<ValueVector> positionEmbeddings;

  // The main Transformer Encoder backbone
  final TransformerEncoder transformerEncoder;

  // Final MLP head for classification
  final Layer mlpHead;

  VideoTransformer({
    required this.frameEmbedDim,
    required this.embedSize, // Often embedSize == frameEmbedDim
    required this.maxVideoSequenceLength,
    required this.numClasses,
    this.numLayers = 2, // Reduced for faster example execution
    this.numHeads = 4, // Reduced for faster example execution
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        // Only create projection if dimensions don't match
        frameProjection = (frameEmbedDim != embedSize)
            ? Layer.fromNeurons(frameEmbedDim, embedSize)
            : null,
        // Position embeddings for the sequence of video frame embeddings
        positionEmbeddings = List.generate(
            maxVideoSequenceLength,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // The TransformerEncoder processes the embedded frames
        transformerEncoder = TransformerEncoder(
          vocabSize: 0, // Not used directly
          embedSize: embedSize,
          blockSize: maxVideoSequenceLength,
          numLayers: numLayers,
          numHeads: numHeads,
        ),
        // Classification head
        mlpHead = Layer.fromNeurons(embedSize, numClasses);

  /// The forward pass for the Video Transformer.
  ///
  /// Takes a list of pre-extracted video frame/clip embeddings.
  /// Returns logits for video classification.
  List<Value> forward(List<ValueVector> videoEmbeddings) {
    if (videoEmbeddings.length > maxVideoSequenceLength) {
      throw ArgumentError(
          "Video sequence length (${videoEmbeddings.length}) exceeds "
          "maxVideoSequenceLength ($maxVideoSequenceLength).");
    }

    // 1. Optional: Project frame embeddings if dimensions don't match Transformer's embedSize
    final projectedEmbeddings = frameProjection != null
        ? videoEmbeddings.map((e) => frameProjection!.forward(e)).toList()
        : videoEmbeddings;

    // 2. Add positional embeddings
    final sequenceWithPositionalEmbeddings =
        List.generate(projectedEmbeddings.length, (i) {
      return projectedEmbeddings[i] + positionEmbeddings[i];
    });

    // 3. Pass the sequence through the Transformer Encoder
    final encodedFeatures =
        transformerEncoder.forwardEmbeddings(sequenceWithPositionalEmbeddings);

    // 4. Global Average Pooling over the sequence to get a single vector for classification
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
    final params = <Value>[];
    if (frameProjection != null) {
      params.addAll(frameProjection!.parameters());
    }
    params.addAll(positionEmbeddings.expand((vec) => vec.values));
    params.addAll(transformerEncoder.parameters());
    params.addAll(mlpHead.parameters());
    return params;
  }
}
