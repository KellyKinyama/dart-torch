import '../tensor/tensor.dart';
import 'layer.dart';
import 'layer_norm.dart';
import 'module.dart';
import 'transformer_decoder_block.dart';

class TransformerDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int encoderEmbedSize;

  // Learnable Tensors
  final Tensor tokenEmbeddingTable;
  final Tensor positionEmbeddingTable;

  final List<TransformerDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  TransformerDecoder({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.encoderEmbedSize = 64,
    int numLayers = 4,
    int numHeads = 4,
  })  : tokenEmbeddingTable = Tensor.random([vocabSize, embedSize]),
        positionEmbeddingTable = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(
          numLayers,
          (_) => TransformerDecoderBlock(embedSize, numHeads, encoderEmbedSize),
        ),
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer(embedSize, vocabSize, useGelu: false);

  Tensor forward(List<int> idx, Tensor encoderOutput) {
    final int T = idx.length;

    // 1. Create x and MANUALLY link it to the tables
    final x = Tensor([T, embedSize],
        children: {tokenEmbeddingTable, positionEmbeddingTable});

    for (int t = 0; t < T; t++) {
      int tokenIdx = idx[t];
      for (int i = 0; i < embedSize; i++) {
        x.data[t * embedSize + i] =
            tokenEmbeddingTable.data[tokenIdx * embedSize + i] +
                positionEmbeddingTable.data[t * embedSize + i];
      }
    }

    // This function is the ONLY way gradients get from the Transformer back to the weights
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

    // 2. Transformer Blocks
    Tensor hidden = x;
    for (var block in blocks) {
      hidden = block.forward(hidden, encoderOutput);
    }

    // 3. Final projection
    Tensor normalized = finalLayerNorm.forward(hidden);
    return lmHead.forward(normalized);
  }

  @override
  List<Tensor> parameters() => [
        tokenEmbeddingTable,
        positionEmbeddingTable,
        ...blocks.expand((b) => b.parameters()),
        ...finalLayerNorm.parameters(),
        ...lmHead.parameters(),
      ];
}
