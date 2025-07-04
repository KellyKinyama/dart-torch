// file: feed_forward.dart

import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A standard two-layer feed-forward network (FFN).
class FeedForward extends Module {
  final Layer layer1;
  final Layer layer2;

  // The FFN expands the embedding size and then contracts it back.
  FeedForward(int embedSize)
      : layer1 = Layer.fromNeurons(embedSize, 4 * embedSize),
        layer2 = Layer.fromNeurons(4 * embedSize, embedSize);

  /// Forward pass for the FFN.
  ValueVector forward(ValueVector x) {
    // Apply: (Linear -> ReLU -> Linear)
    final x1 = layer1.forward(x);
    final x_relu = x1.reLU();
    final x2 = layer2.forward(x_relu);
    return x2;
  }

  @override
  List<Value> parameters() {
    return [
      ...layer1.parameters(),
      ...layer2.parameters(),
    ];
  }
}
