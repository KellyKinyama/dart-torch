import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class AFTCrossAttention extends Module {
  final Layer keyLayer; // Encoder projection
  final Layer queryLayer; // Decoder projection
  final Layer valueLayer; // Encoder projection
  final int headSize;
  final Tensor posBias; // [maxTDec, maxTEnc]

  AFTCrossAttention(int decoderEmbedSize, int encoderEmbedSize, this.headSize,
      int maxTDec, int maxTEnc)
      : keyLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
        queryLayer = Layer(decoderEmbedSize, headSize, useGelu: false),
        valueLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
        posBias = Tensor.random([maxTDec, maxTEnc]);

  /// xDec: [TDec, decoderEmbedSize]
  /// xEnc: [TEnc, encoderEmbedSize]
  Tensor forward(Tensor xDec, Tensor xEnc) {
    final int TDec = xDec.shape[0];
    final int TEnc = xEnc.shape[0];
    final int d = headSize;

    // 1. Projections
    final Q = queryLayer.forward(xDec).sigmoid(); // Decoder query
    final K = keyLayer.forward(xEnc); // Encoder keys
    final V = valueLayer.forward(xEnc); // Encoder values

    final out = Tensor([TDec, d], children: {Q, K, V, posBias});

    // 2. Cross-Attention Loop
    for (int t = 0; t < TDec; t++) {
      Tensor numerator = Tensor.zeros([1, d]);
      Tensor denominator = Tensor.zeros([1, d]);

      // Decoder position 't' attends to all Encoder positions 'tp'
      for (int tp = 0; tp < TEnc; tp++) {
        // Index into the rectangular position bias matrix
        double w_ttp = posBias.data[t * posBias.shape[1] + tp];

        // exp(K_tp + w_ttp)
        final expWeight = (K.getRow(tp) + w_ttp).exp();

        numerator = numerator + (expWeight * V.getRow(tp));
        denominator = denominator + expWeight;
      }

      // Final combination for this decoder step
      final row_t = Q.getRow(t) * (numerator / (denominator + 1e-9));
      out.setRow(t, row_t);
    }

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...keyLayer.parameters(),
        ...queryLayer.parameters(),
        ...valueLayer.parameters(),
        posBias
      ];
}
