import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer/aft_transformer_encoder.dart';

/// ViT-like spectrogram refiner:
/// Input: flattened spectrogram of shape [timeSteps * freqBins]
/// Output: refined frames List<ValueVector> of shape [timeSteps][freqBins]
class ViTSpectrogramRefiner extends Module {
  final int timeSteps; // e.g. 64
  final int freqBins; // e.g. 513
  final int patchTime; // e.g. 4 (patch spans 4 frames)
  final int embedSize; // MUST be divisible by numHeads (e.g. 512)
  final int numLayers;
  final int numHeads;

  final Layer patchProjection; // (freqBins*patchTime) -> embedSize
  final Layer outputProjection; // embedSize -> (freqBins*patchTime)

  final List<ValueVector> positionEmbeddings; // length = timeSteps/patchTime
  final TransformerEncoder encoder;

  ViTSpectrogramRefiner({
    required this.timeSteps,
    required this.freqBins,
    this.patchTime = 4,
    required this.embedSize,
    this.numLayers = 2,
    this.numHeads = 4,
  })  : assert(timeSteps % patchTime == 0,
            "timeSteps must be divisible by patchTime"),
        assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        patchProjection = Layer.fromNeurons(freqBins * patchTime, embedSize),
        outputProjection = Layer.fromNeurons(embedSize, freqBins * patchTime),
        positionEmbeddings = List.generate(
          timeSteps ~/ patchTime,
          (_) => ValueVector.fromDoubleList(
            List.generate(
                embedSize, (_) => math.Random().nextDouble() * 0.02 - 0.01),
          ),
        ),
        encoder = TransformerEncoder(
          vocabSize: 0, // embeddings are provided directly
          embedSize: embedSize,
          blockSize: timeSteps ~/ patchTime,
          numLayers: numLayers,
          numHeads: numHeads,
        );

  List<ValueVector> _makePatches(List<double> flatSpec) {
    final expected = timeSteps * freqBins;
    if (flatSpec.length != expected) {
      throw ArgumentError(
          "flatSpec length ${flatSpec.length} != expected $expected "
          "(timeSteps=$timeSteps, freqBins=$freqBins)");
    }

    final patches = <ValueVector>[];

    for (int t = 0; t < timeSteps; t += patchTime) {
      final patch = <double>[];

      for (int dt = 0; dt < patchTime; dt++) {
        final base = (t + dt) * freqBins;
        for (int f = 0; f < freqBins; f++) {
          patch.add(flatSpec[base + f]);
        }
      }

      patches.add(ValueVector.fromDoubleList(patch));
    }

    return patches;
  }

  List<ValueVector> _unpatch(List<ValueVector> encodedPatches) {
    final frames = <ValueVector>[];

    for (final patchEmbed in encodedPatches) {
      // project embedding back to patch-space (freqBins*patchTime)
      final projected = outputProjection.forward(patchEmbed);
      final vals = projected.values;

      // safety check
      final expectedPatchSize = freqBins * patchTime;
      if (vals.length != expectedPatchSize) {
        throw StateError(
            "Projected patch length ${vals.length} != expected $expectedPatchSize");
      }

      int idx = 0;
      for (int dt = 0; dt < patchTime; dt++) {
        final frameVals = <Value>[];
        for (int f = 0; f < freqBins; f++) {
          frameVals.add(vals[idx++]);
        }
        frames.add(ValueVector(frameVals));
      }
    }

    // safety check
    if (frames.length != timeSteps) {
      throw StateError(
          "Unpatched frames ${frames.length} != timeSteps $timeSteps");
    }

    return frames;
  }

  /// Forward returns refined frames [timeSteps][freqBins]
  List<ValueVector> forward(List<double> flatSpec) {
    final patches = _makePatches(flatSpec);

    // patch -> embedding + position
    final embedded = List.generate(patches.length, (i) {
      final pe = positionEmbeddings[i];
      return patchProjection.forward(patches[i]) + pe;
    });

    final encoded = encoder.forwardEmbeddings(embedded);
    return _unpatch(encoded);
  }

  @override
  List<Value> parameters() {
    return [
      ...patchProjection.parameters(),
      ...outputProjection.parameters(),
      ...positionEmbeddings.expand((v) => v.values),
      ...encoder.parameters(),
    ];
  }
}
