import '../tensor/tensor.dart';
import 'transformer_encoder_block.dart';
import 'layer_norm.dart';
import 'module.dart';

class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;

  // Learnable Tensors (Continuous memory buffers)
  final Tensor tokenEmbeddingTable;
  final Tensor positionEmbeddingTable;

  final List<TransformerEncoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100,
    this.embedSize = 64,
    this.blockSize = 128,
    int numLayers = 6,
    int numHeads = 8,
  })  : assert(embedSize % numHeads == 0),
        // Shape: [Vocab, Embed]
        tokenEmbeddingTable = Tensor.random([vocabSize, embedSize]),
        // Shape: [MaxSequence, Embed]
        positionEmbeddingTable = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(
          numLayers,
          (_) => TransformerEncoderBlock(embedSize, numHeads),
        ),
        finalLayerNorm = LayerNorm(embedSize);

  /// Standard NLP Forward: Input is a list of token indices
  Tensor forward(List<int> idx) {
    final int T = idx.length;
    if (T > blockSize) throw ArgumentError("Sequence exceeds block size");

    // 1. Create a fresh Tensor for combined embeddings [T, embedSize]
    final x = Tensor([T, embedSize]);

    for (int t = 0; t < T; t++) {
      int tokenIdx = idx[t];
      for (int i = 0; i < embedSize; i++) {
        // Sum token + position embedding in one go
        x.data[t * embedSize + i] =
            tokenEmbeddingTable.data[tokenIdx * embedSize + i] +
                positionEmbeddingTable.data[t * embedSize + i];
      }
    }

    // Manual backward for the lookup/sum step
    x.onBackward = () {
      for (int t = 0; t < T; t++) {
        int tokenIdx = idx[t];
        for (int i = 0; i < embedSize; i++) {
          double g = x.grad[t * embedSize + i];
          tokenEmbeddingTable.grad[tokenIdx * embedSize + i] += g;
          positionEmbeddingTable.grad[t * embedSize + i] += g;
        }
      }
    };

    return _forwardThroughBlocks(x);
  }

  /// Vision/Multi-modal Forward: Input is already pre-computed embeddings
  Tensor forwardEmbeddings(Tensor embeddedInputs) {
    if (embeddedInputs.shape[0] > blockSize) {
      throw ArgumentError("Sequence length exceeds block size");
    }
    return _forwardThroughBlocks(embeddedInputs);
  }

  /// Internal processing: Blocks + Final Norm
  Tensor _forwardThroughBlocks(Tensor x) {
    Tensor hidden = x;
    for (var block in blocks) {
      hidden = block.forward(hidden);
    }
    return finalLayerNorm.forward(hidden);
  }

  @override
  List<Tensor> parameters() => [
        tokenEmbeddingTable,
        positionEmbeddingTable,
        ...blocks.expand((b) => b.parameters()),
        ...finalLayerNorm.parameters(),
      ];
}
