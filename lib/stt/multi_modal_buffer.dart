import 'dart:typed_data';
import '/nn/value.dart';
import '/nn/value_vector.dart';

class MultimodalBuffer {
  /// Converts raw Mel-Spectrogram frames (dB) into normalized ValueVectors.
  /// 
  /// [spectrogram] - List of frames from your powerToDb function.
  /// [topDb] - The floor used during DB conversion (e.g., 80.0).
  static List<ValueVector> prepareAudio(
    List<Float64List> spectrogram, {
    required int maxLen,
    double topDb = 80.0,
  }) {
    // 1. Limit sequence length to model's max capacity
    final frames = spectrogram.length > maxLen 
        ? spectrogram.sublist(0, maxLen) 
        : spectrogram;

    return frames.map((frame) {
      final normalizedValues = frame.map((db) {
        // Normalize dB from [-80, 0] to [0, 1] range for better SGD convergence
        double norm = (db + topDb) / topDb;
        return Value(norm.clamp(0.0, 1.0));
      }).toList();
      
      return ValueVector(normalizedValues);
    }).toList();
  }

  /// Prepares Video Embeddings for the transformer.
  static List<ValueVector> prepareVideo(
    List<List<double>> embeddings, {
    required int maxLen,
  }) {
    final frames = embeddings.length > maxLen 
        ? embeddings.sublist(0, maxLen) 
        : embeddings;

    return frames.map((e) => ValueVector.fromDoubleList(e)).toList();
  }
}