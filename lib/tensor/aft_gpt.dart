import 'aft_transformer_decoder_block.dart';
import 'embedding.dart';
import 'layer_norm.dart';
import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class AFT_GPT extends Module {
  final Embedding tokenEmbed;
  final Tensor posEmbed;
  final List<TransformerDecoderBlock> blocks;
  final LayerNorm finalLn;
  final Linear lmHead; // Language Model Head
  final int blockSize;

  AFT_GPT({
    required int vocabSize,
    required int embedSize,
    required this.blockSize,
    required int numLayers,
    required int numHeads,
  })  : tokenEmbed = Embedding(vocabSize, embedSize),
        posEmbed = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(
            numLayers,
            (_) => TransformerDecoderBlock(
                embedSize, numHeads, embedSize, blockSize)),
        finalLn = LayerNorm(embedSize),
        lmHead = Linear(embedSize, vocabSize);

  /// Standard Generative Forward
  Tensor forward(List<int> idx, [Tensor? encoderHiddenStates]) {
    final T = idx.length;

    // 1. Token + Position Embeddings
    Tensor x = tokenEmbed.forward(idx) + posEmbed.slice(0, T);

    // 2. Transformer Blocks
    for (var block in blocks) {
      // If doing pure GPT (Decoder-only), x_enc can be same as x or null
      x = block.forward(x, encoderHiddenStates ?? x);
    }

    // 3. Final Norm and Head
    x = finalLn.forward(x);
    return lmHead.forward(x); // Returns [T, VocabSize] Logits
  }

  @override
  List<Tensor> parameters() => [
        ...tokenEmbed.parameters(),
        posEmbed,
        ...blocks.expand((b) => b.parameters()),
        ...finalLn.parameters(),
        ...lmHead.parameters(),
      ];
}
