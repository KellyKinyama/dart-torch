import 'dart:io';
import 'dart:convert';
import '/nn/module.dart'; // Assuming module.dart is in the same directory or accessible via path
import '/nn/value.dart'; // Assuming value.dart is in the same directory or accessible via path

/// Saves the parameters (weights) of a given Module to a JSON file.
///
/// The parameters are extracted as a list of double values from the
/// `Module.parameters()` method and then encoded as a JSON array.
///
/// [module]: The Module instance whose parameters are to be saved.
/// [filePath]: The path to the file where the weights will be saved.
/// Saves the parameters (weights) of a given Module to a JSON file.
///
/// The parameters are extracted as a list of double values from the
/// `Module.parameters()` method and then encoded as a JSON array.
///
/// [module]: The Module instance whose parameters are to be saved.
/// [filePath]: The path to the file where the weights will be saved.
Future<void> saveModuleParameters(Module module, String filePath) async {
  final List<Value> parameters = module.parameters();
  final List<double> weightsData = parameters.map((v) => v.data).toList();

  final String jsonString = jsonEncode(weightsData);
  final File file = File(filePath);

  try {
    await file.writeAsString(jsonString);
    print('Weights successfully saved to: $filePath');
  } catch (e) {
    print('Error saving weights to $filePath: $e');
  }
}

/// Loads parameters (weights) from a JSON file into a given Module.
///
/// The function reads a JSON array of doubles from the specified file
/// and then iterates through the Module's parameters, updating their `data` field.
/// It's crucial that the order of parameters when saving matches the order when loading.
///
/// [module]: The Module instance whose parameters are to be loaded.
/// [filePath]: The path to the file from which the weights will be loaded.
Future<void> loadModuleParameters(Module module, String filePath) async {
  final File file = File(filePath);

  if (!await file.exists()) {
    print('Error: File not found at $filePath');
    return;
  }

  try {
    final String jsonString = await file.readAsString();
    final List<dynamic> loadedData =
        jsonDecode(jsonString); // jsonDecode returns List<dynamic>

    final List<Value> moduleParameters = module.parameters();

    if (loadedData.length != moduleParameters.length) {
      print('Warning: Mismatch in number of parameters. '
          'Loaded: ${loadedData.length}, Expected: ${moduleParameters.length}');
    }

    // Update the data of each Value object in the module's parameters
    for (int i = 0; i < moduleParameters.length && i < loadedData.length; i++) {
      // Explicitly convert the dynamic value to double
      if (loadedData[i] is num) {
        moduleParameters[i].setData((loadedData[i] as num).toDouble());
      } else {
        print(
            'Error: Expected a number at index $i, but got ${loadedData[i].runtimeType}');
        // You might want to handle this error more robustly, e.g., throw an exception
      }
    }
    print('Weights successfully loaded from: $filePath');
  } catch (e) {
    print('Error loading weights from $filePath: $e');
  }
}
