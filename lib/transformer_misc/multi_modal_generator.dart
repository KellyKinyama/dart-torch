import '../transformer/transformer_decoder.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/module.dart';
import '/nn/layer.dart';
import 'audio_transformer.dart';
import 'aft_video_transformer.dart';

class MultimodalGenerator extends Module {
  final AudioTransformer audioEncoder;
  final VideoTransformer videoEncoder;
  final TransformerDecoder decoder;

  MultimodalGenerator({
    required this.audioEncoder,
    required this.videoEncoder,
    required this.decoder,
  });

  /// [textTokenIds] is now correctly passed as a List<int>
  List<ValueVector> forward(List<ValueVector> audioFeatures,
      List<ValueVector> videoEmbeddings, List<int> textTokenIds) {
    // 1. Encode Audio (Accessing the projection and encoder logic)
    final projectedAudio = audioEncoder.project(audioFeatures);
    final encodedAudio = audioEncoder.transformerEncoder.forwardEmbeddings(
        _applyPositional(projectedAudio, audioEncoder.positionEmbeddings));

    // 2. Encode Video
    final projectedVideo = videoEncoder.project(videoEmbeddings);
    final encodedVideo = videoEncoder.transformerEncoder.forwardEmbeddings(
        _applyPositional(projectedVideo, videoEncoder.positionEmbeddings));

    // 3. Fusion Context
    final multimodalContext = [...encodedAudio, ...encodedVideo];

    // 4. Decode (Matches the decoder.forward(List<int>, List<ValueVector>) signature)
    // This will return the logits for text generation
    return decoder.forward(textTokenIds, multimodalContext);
  }

  List<ValueVector> _applyPositional(
      List<ValueVector> input, List<ValueVector> pos) {
    return List.generate(input.length, (i) => input[i] + pos[i]);
  }

  @override
  List<Value> parameters() => [
        ...audioEncoder.parameters(),
        ...videoEncoder.parameters(),
        ...decoder.parameters(),
      ];
}
