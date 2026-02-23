import '../tensor/tensor.dart';
import 'aft_transformer_decoder_block.dart';
import 'layer_norm.dart';
import 'layer.dart';
import 'module.dart';

class TransformerDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;
  final int encoderEmbedSize;

  // Embedding Tensors
  final Tensor wte; // Weight Token Embeddings [vocabSize, embedSize]
  final Tensor wpe; // Weight Position Embeddings [blockSize, embedSize]

  final List<TransformerDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  TransformerDecoder({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
    this.encoderEmbedSize = 64,
  })  : wte = Tensor.random([vocabSize, embedSize]),
        wpe = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(
            numLayers,
            (i) => TransformerDecoderBlock(
                  embedSize,
                  numHeads,
                  encoderEmbedSize,
                  blockSize,
                )),
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer(embedSize, vocabSize, useGelu: false) {
    // Scaling embeddings down for stability
    for (int i = 0; i < wte.data.length; i++) wte.data[i] *= 0.02;
    for (int i = 0; i < wpe.data.length; i++) wpe.data[i] *= 0.02;
  }

  /// idx: List of token IDs
  /// encoderOutput: Tensor of shape [T_enc, encoderEmbedSize]
  Tensor forward(List<int> idx, Tensor encoderOutput) {
    final int T = idx.length;
    if (T > blockSize) {
      throw ArgumentError("Sequence length $T > block size $blockSize");
    }

    // 1. Embeddings: Get token and position rows and sum them
    // Output shape: [T, embedSize]
    Tensor x = Tensor.zeros([T, embedSize]);
    for (int t = 0; t < T; t++) {
      final tok_emb = wte.getRow(idx[t]);
      final pos_emb = wpe.getRow(t);
      x.setRow(t, tok_emb + pos_emb);
    }

    // 2. Transformer Blocks
    // Data flows through the entire stack
    for (final block in blocks) {
      x = block.forward(x, encoderOutput);
    }

    // 3. Final Layer Norm
    x = finalLayerNorm.forward(x);

    // 4. Output Head (Logits)
    // Result shape: [T, vocabSize]
    return lmHead.forward(x);
  }

  @override
  List<Tensor> parameters() => [
        wte,
        wpe,
        ...blocks.expand((block) => block.parameters()),
        ...finalLayerNorm.parameters(),
        ...lmHead.parameters(),
      ];
}


