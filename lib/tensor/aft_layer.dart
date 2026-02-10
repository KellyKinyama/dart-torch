import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class AFTLayer extends Module {
  final Linear qProj;
  final Linear kProj;
  final Linear vProj;
  final Tensor posBias; // Learned relative position bias

  AFTLayer(int dim, int seqLen)
      : qProj = Linear(dim, dim),
        kProj = Linear(dim, dim),
        vProj = Linear(dim, dim),
        posBias = Tensor.random([seqLen, seqLen]);

  @override
  Tensor forward(Tensor x) {
    // x shape: [Batch, SeqLen, Dim]
    // For simplicity in this engine, let's treat it as [SeqLen, Dim]

    final Q = qProj.forward(x).sigmoid(); // Gating
    final K = kProj.forward(x).exp(); // Importance
    final V = vProj.forward(x); // Information

    // AFT-Full weighting logic
    // We use the learned position bias to weight how much each
    // time-step cares about others.
    final weight = K.matmul(posBias);
    final context = (K * V).matmul(posBias);

    // Output = Sigmoid(Q) * (WeightedValues / WeightsSum)
    return Q * (context / weight);
  }

  @override
  List<Tensor> parameters() => [
        ...qProj.parameters(),
        ...kProj.parameters(),
        ...vProj.parameters(),
        posBias
      ];
}
