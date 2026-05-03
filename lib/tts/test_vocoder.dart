import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../stft_spectrogram.dart';
import 'simple_vocoder.dart';
import 'pitch_vocoder.dart';
import 'package:audio_codec/src/wav/wav_encoder.dart';

void main() async {
  print("---- TESTING VOCODER ONLY ----");

  const sampleRate = 16000;
  const maxAudioLen = 32;

  // ✅ 1. Load real STFT
  final spectrogram = await stftSpectrogram("output.wav");

  // ✅ 2. Use ONLY real magnitudes (NO NN)
  final magnitudes = spectrogram
      .take(maxAudioLen)
      .map((frame) => Float64List.fromList(
            frame.map((v) {
              // ensure positive magnitude
              return v.clamp(0.0, 1000.0);
            }).toList(),
          ))
      .toList();

  print("Frames: ${magnitudes.length}");
  print("Bins: ${magnitudes[0].length}");

  // ✅ 3. Try BOTH vocoders
  final simpleVocoder = SimpleVocoder(
    frameSize: 1024,
    hopSize: 256,
  );

  final pitchVocoder = PitchVocoder(
    frameSize: 1024,
    hopSize: 256,
  );

  // ✅ 4. Generate audio
  final pcm1 = simpleVocoder.generate(magnitudes, sampleRate);
  final pcm2 = pitchVocoder.generate(magnitudes, sampleRate);

  // ✅ 5. Save both outputs
  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  encoder.encode(File("test_simple.wav"), pcm1);
  encoder.encode(File("test_pitch.wav"), pcm2);

  print("✅ Saved:");
  print(" - test_simple.wav");
  print(" - test_pitch.wav");
}
