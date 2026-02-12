import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class LMHead extends Module {
  final Layer projection;

  LMHead(int embedSize, int vocabSize)
      : projection = Layer(embedSize, vocabSize, useGelu: false);

  Tensor forward(Tensor x) {
    // x: [SequenceLength, EmbedSize]
    // returns: [SequenceLength, VocabSize]
    return projection.forward(x);
  }

  @override
  List<Tensor> parameters() => projection.parameters();
}
