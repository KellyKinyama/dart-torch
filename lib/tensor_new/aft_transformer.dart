import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'aft_transformer_block.dart'; // Ensure this uses your Tensor-based block
import 'layer_norm.dart';
import 'layer.dart';
import 'module.dart';

/// A complete decoder-only Transformer model (GPT-style) using the Tensor engine.
class Transformer extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize; 
  final int numLayers;
  final int numHeads;

  // Embedding Tensors
  final Tensor wte; // Token embeddings [vocabSize, embedSize]
  final Tensor wpe; // Position embeddings [blockSize, embedSize]

  final List<TransformerBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  Transformer({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
  })  : wte = Tensor.random([vocabSize, embedSize]),
        wpe = Tensor.random([blockSize, embedSize]),
        blocks = List.generate(
            numLayers,
            (i) => TransformerBlock(embedSize, numHeads, blockSize, masked: true)),
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer(embedSize, vocabSize, useGelu: false) {
    // Standard initialization scaling
    for (int i = 0; i < wte.data.length; i++) wte.data[i] *= 0.02;
    for (int i = 0; i < wpe.data.length; i++) wpe.data[i] *= 0.02;
  }

  /// Forward pass takes a sequence of indices and returns [T, vocabSize] logits.
  Tensor forward(List<int> idx) {
    final int T = idx.length;
    if (T > blockSize) {
      throw ArgumentError("Sequence length $T exceeds block size $blockSize");
    }

    // 1. Embedding Lookup and Summation
    // Shape: [T, embedSize]
    Tensor x = Tensor.zeros([T, embedSize]);
    for (int t = 0; t < T; t++) {
      final tokEmb = wte.getRow(idx[t]);
      final posEmb = wpe.getRow(t);
      x.setRow(t, tokEmb + posEmb);
    }

    // 2. Transformer Blocks
    // Data flows through the stack; each block uses AFT and causal masking
    for (final block in blocks) {
      x = block.forward(x);
    }

    // 3. Final Layer Normalization
    x = finalLayerNorm.forward(x);

    // 4. LM Head (Projection to Vocab)
    // Output Shape: [T, vocabSize]
    return lmHead.forward(x);
  }

  @override
  List<Tensor> parameters() => [
        wte,
        wpe,
        ...blocks.expand((b) => b.parameters()),
        ...finalLayerNorm.parameters(),
        ...lmHead.parameters(),
      ];
}