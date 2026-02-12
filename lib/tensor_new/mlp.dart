import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class MLP extends Module {
  final List<Layer> layers = [];

  MLP(int nin, List<int> nouts) {
    List<int> dims = [nin, ...nouts];
    for (int i = 0; i < nouts.length; i++) {
      // The last layer usually shouldn't have ReLU if it's for regression/logits
      bool isLast = i == nouts.length - 1;
      layers.add(Layer(dims[i], dims[i + 1], useGelu: !isLast));
    }
  }

  Tensor forward(Tensor x) {
    Tensor out = x;
    for (var layer in layers) {
      out = layer.forward(out);
    }
    return out;
  }

  @override
  List<Tensor> parameters() => layers.expand((l) => l.parameters()).toList();
}

void main() {
  // 1. Setup Network (2 inputs, 4 hidden, 1 output)
  final model = MLP(2, [4, 1]);
  final double learningRate = 0.01;

  // 2. Mock Data (XOR-ish problem)
  final x = Tensor([4, 2]);
  x.data.setAll(0, [0, 0, 0, 1, 1, 0, 1, 1]);

  final target = Tensor([4, 1]);
  target.data.setAll(0, [0, 1, 1, 0]);

  // 3. Training Loop
  for (int epoch = 0; epoch < 100; epoch++) {
    // Forward
    final pred = model.forward(x);
    final loss = pred.mseLoss(target);

    // Backward
    model.zeroGrad();
    loss.backward();

    // Update (SGD Step)
    for (var p in model.parameters()) {
      for (int i = 0; i < p.length; i++) {
        p.data[i] -= learningRate * p.grad[i];
      }
    }

    if (epoch % 10 == 0) print("Epoch $epoch, Loss: ${loss.data[0]}");
  }
}
