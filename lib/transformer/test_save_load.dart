// main.dart or a test file
import 'encoder_decoder_transformer.dart';
import 'network_utils.dart'; // Import the new utility file
// import '/nn/value.dart';
// import '/nn/value_vector.dart';

void main() async {
  // 1. Create a dummy Transformer model
  final transformer = EncoderDecoderTransformer(
    sourceVocabSize: 100,
    targetVocabSize: 100,
    embedSize: 32,
    sourceBlockSize: 10,
    targetBlockSize: 10,
    numLayers: 2,
    numHeads: 4,
  );

  print('Initial parameters before saving:');
  final initialParams = transformer.parameters();
  // Print first few parameters to verify
  for (int i = 0;
      i < (initialParams.length > 5 ? 5 : initialParams.length);
      i++) {
    print('Param $i: ${initialParams[i].data}');
  }
  print('Total initial parameters: ${initialParams.length}');

  final String filePath = 'transformer_weights.json';

  // 2. Save the weights to a file
  await saveModuleParameters(transformer, filePath);

  // 3. (Optional) Modify weights or create a new model instance
  // For demonstration, let's just re-initialize the transformer to simulate
  // loading into a fresh model or after training.
  final newTransformer = EncoderDecoderTransformer(
    sourceVocabSize: 100,
    targetVocabSize: 100,
    embedSize: 32,
    sourceBlockSize: 10,
    targetBlockSize: 10,
    numLayers: 2,
    numHeads: 4,
  );

  print('\nParameters of new transformer before loading:');
  final newParamsBeforeLoad = newTransformer.parameters();
  for (int i = 0;
      i < (newParamsBeforeLoad.length > 5 ? 5 : newParamsBeforeLoad.length);
      i++) {
    print('Param $i: ${newParamsBeforeLoad[i].data}');
  }

  // 4. Load the weights from the file into the new model
  await loadModuleParameters(newTransformer, filePath);

  print('\nParameters of new transformer after loading:');
  final newParamsAfterLoad = newTransformer.parameters();
  for (int i = 0;
      i < (newParamsAfterLoad.length > 5 ? 5 : newParamsAfterLoad.length);
      i++) {
    print('Param $i: ${newParamsAfterLoad[i].data}');
  }

  // Verify that the loaded weights are the same as the initial saved weights
  bool weightsMatch = true;
  if (initialParams.length != newParamsAfterLoad.length) {
    weightsMatch = false;
  } else {
    for (int i = 0; i < initialParams.length; i++) {
      if (initialParams[i].data != newParamsAfterLoad[i].data) {
        weightsMatch = false;
        break;
      }
    }
  }
  print('\nDo initial and loaded weights match? $weightsMatch');

  // Example of using the transformer for a forward pass (requires actual implementation for sub-modules)
  // try {
  //   List<int> sourceInput = List.generate(10, (index) => index % 100);
  //   List<int> targetInput = List.generate(8, (index) => index % 100);
  //   final logits = newTransformer.forward(sourceInput, targetInput);
  //   print('Forward pass successful. Logits length: ${logits.length}');
  // } catch (e) {
  //   print('Error during forward pass: $e');
  // }
}
