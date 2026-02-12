import '../tensor/tensor.dart';
import 'aft_transformer_encoder_block.dart';
import 'layer_norm.dart';
import 'module.dart';

/// A complete Transformer Encoder model (Tensor-based).
class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;

  // Embedding Tensors
  final Tensor wte; // Weight Token Embeddings [vocabSize, embedSize]
  final Tensor wpe; // Weight Position Embeddings [blockSize, embedSize]

  final List<TransformerEncoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100,
    this.embedSize = 64,
    this.blockSize = 128,
    this.numLayers = 6,
    this.numHeads = 8,
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        wte = Tensor.random([vocabSize, embedSize]),
        wpe = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(numLayers,
            (i) => TransformerEncoderBlock(embedSize, numHeads, blockSize)),
        finalLayerNorm = LayerNorm(embedSize) {
    // Standard initialization scaling for stability
    for (int i = 0; i < wte.data.length; i++) wte.data[i] *= 0.02;
    for (int i = 0; i < wpe.data.length; i++) wpe.data[i] *= 0.02;
  }

  /// idx: List of token IDs. Returns [T, embedSize] Tensor of contextual embeddings.
  Tensor forward(List<int> idx) {
    final int T = idx.length;
    if (T > blockSize) {
      throw ArgumentError("Sequence length $T exceeds block size $blockSize");
    }

    // 1. Combine token and position embeddings
    // We create a new [T, embedSize] Tensor for the current sequence
    Tensor x = Tensor.zeros([T, embedSize]);
    for (int t = 0; t < T; t++) {
      // getRow provides a slice that tracks gradients back to the embedding tables
      final tokEmb = wte.getRow(idx[t]);
      final posEmb = wpe.getRow(t);
      x.setRow(t, tokEmb + posEmb);
    }

    return _processThroughBlocks(x);
  }

  /// For cases where inputs are already vectors (like image patches in ViT)
  Tensor forwardEmbeddings(Tensor embeddedInputs) {
    if (embeddedInputs.shape[0] > blockSize) {
      throw ArgumentError("Input sequence length exceeds block size");
    }
    return _processThroughBlocks(embeddedInputs);
  }

  // Internal processing pipeline
  Tensor _processThroughBlocks(Tensor x) {
    // 2. Pass sequence through the stack of AFT Encoder Blocks
    for (final block in blocks) {
      x = block.forward(x);
    }

    // 3. Final Layer Norm
    return finalLayerNorm.forward(x);
  }

  @override
  List<Tensor> parameters() => [
        wte,
        wpe,
        ...blocks.expand((block) => block.parameters()),
        ...finalLayerNorm.parameters(),
      ];
}
