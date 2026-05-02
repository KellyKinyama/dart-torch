import 'dart:math' as math;
import 'dart:typed_data';

class GriffinLimGenerator {
  final int iterations;
  final int frameSize;
  final int hopSize;

  GriffinLimGenerator({
    this.iterations = 60,
    this.frameSize = 1024,
    this.hopSize = 256,
  });

  Uint8List generateWav(List<Float64List> linearSpectrogram, int sampleRate) {
    int numFrames = linearSpectrogram.length;
    int totalSamples = numFrames * hopSize + frameSize;

    // 1. Initialize with random Phase
    List<Float64List> phases = List.generate(
        numFrames,
        (_) => Float64List(frameSize ~/ 2 + 1)
            .map((_) => math.Random().nextDouble() * 2 * math.pi)
            .toList() as Float64List);

    Float64List signal = Float64List(totalSamples);

    // 2. Iterative Reconstruction
    for (int iter = 0; iter < iterations; iter++) {
      signal = _inverseSTFT(linearSpectrogram, phases);
      phases = _forwardSTFTPhase(signal, numFrames);
    }

    // 3. Final normalization and conversion to PCM bytes
    return _convertToWav(signal, sampleRate);
  }

  Float64List _inverseSTFT(List<Float64List> mags, List<Float64List> phases) {
    final output = Float64List(mags.length * hopSize + frameSize);
    final window = _hannWindow(frameSize);

    for (int f = 0; f < mags.length; f++) {
      for (int t = 0; t < frameSize ~/ 2 + 1; t++) {
        // Reconstruct Real and Imaginary parts:
        // Real = Mag * cos(Phase), Imag = Mag * sin(Phase)
        double re = mags[f][t] * math.cos(phases[f][t]);

        // Simplified IFFT step for CPU efficiency
        for (int n = 0; n < frameSize; n++) {
          double angle = 2 * math.pi * t * n / frameSize;
          output[f * hopSize + n] += (re * math.cos(angle)) * window[n];
        }
      }
    }
    return output;
  }

  Uint8List _convertToWav(Float64List signal, int sampleRate) {
    // 1. Peak Normalization: Find the maximum absolute value
    double maxVal = 0.0;
    for (var sample in signal) {
      if (sample.abs() > maxVal) maxVal = sample.abs();
    }

    // 2. Prepare the PCM byte buffer (16-bit = 2 bytes per sample)
    final Int16List pcmList = Int16List(signal.length);

    if (maxVal > 0) {
      for (int i = 0; i < signal.length; i++) {
        // Normalize and scale to 16-bit range (-32768 to 32767)
        double normalized = signal[i] / maxVal;
        pcmList[i] = (normalized * 32767).toInt();
      }
    }

    // 3. Return as Uint8List bytes for your WavEncoder
    return Uint8List.view(pcmList.buffer);
  }

  List<Float64List> _forwardSTFTPhase(Float64List signal, int numFrames) {
    // Extracting the phase for the next iteration
    // Phase = atan2(Imaginary, Real)
    // ... logic for FFT phase extraction ...
    return [];
  }

  Float64List _hannWindow(int size) => Float64List.fromList(List.generate(
      size, (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1)))));
}
