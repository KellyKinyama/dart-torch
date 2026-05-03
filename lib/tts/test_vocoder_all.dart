import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import '../stft_spectrogram.dart';
import 'simple_vocoder.dart';
import 'pitch_vocoder.dart';
import 'griffin_lim_generator.dart'; // ✅ ADD THIS
import 'package:audio_codec/src/wav/wav_encoder.dart';

void main() async {
  print("---- TESTING VOCODERS (INCLUDING GRIFFIN-LIM) ----");

  const sampleRate = 16000;
  const maxAudioLen = 64; // ✅ slightly longer = better GL quality

  // ✅ 1. Load STFT magnitudes
  final spectrogram = await stftSpectrogram(
    "output.wav",
    frameSize: 1024,
    hopSize: 256,
  );

  final magnitudes = spectrogram
      .take(maxAudioLen)
      .map((frame) => Float64List.fromList(
            frame.map((v) {
              // ✅ ensure valid magnitudes
              return v.isFinite ? v.clamp(1e-6, 1e6) : 1e-6;
            }).toList(),
          ))
      .toList();

  print("Frames: ${magnitudes.length}");
  print("Bins: ${magnitudes[0].length}");

  // ✅ 2. Instantiate vocoders
  final simpleVocoder = SimpleVocoder(
    frameSize: 1024,
    hopSize: 256,
  );

  final pitchVocoder = PitchVocoder(
    frameSize: 1024,
    hopSize: 256,
  );

  final griffin = GriffinLimGenerator(
    iterations: 80, // ✅ key setting
    frameSize: 1024,
    hopSize: 256,
  );

  // ✅ 3. Generate audio
  print("Running Simple Vocoder...");
  final pcmSimple = simpleVocoder.generate(magnitudes, sampleRate);

  print("Running Pitch Vocoder...");
  final pcmPitch = pitchVocoder.generate(magnitudes, sampleRate);

  print("Running Griffin–Lim...");
  final pcmGriffin = griffin.generateWav(magnitudes, sampleRate);

  // ✅ 4. Save all outputs
  final encoder = WavEncoder(
    sampleRate: sampleRate,
    numChannels: 1,
    bitDepth: 16,
  );

  encoder.encode(File("test_simple.wav"), pcmSimple);
  encoder.encode(File("test_pitch.wav"), pcmPitch);
  encoder.encode(File("test_griffin.wav"), pcmGriffin);

  print("\n✅ Saved:");
  print(" - test_simple.wav");
  print(" - test_pitch.wav");
  print(" - test_griffin.wav");
}