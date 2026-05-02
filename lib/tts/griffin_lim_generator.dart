import 'dart:math' as math;
import 'dart:typed_data';

class GriffinLimGenerator {
  final int iterations;
  final int frameSize;
  final int hopSize;

  GriffinLimGenerator({
    this.iterations = 200, // Increased for better voice convergence
    this.frameSize = 1024,
    this.hopSize = 256,
  });

  /// Generates a WAV-compatible byte buffer from Mel-spectrogram output.
  Uint8List generateWav(List<Float64List> modelOutput, int sampleRate) {
    int numFrames = modelOutput.length;
    int targetBins = frameSize ~/ 2 + 1; // 513 bins for 1024 FFT

    // 1. Convert predicted Power Spectrogram to Magnitude Spectrogram
    // and interpolate 32 Mel-bands to 513 Linear bins.
    List<Float64List> magnitudes = modelOutput.map((melFrame) {
      final expanded = Float64List(targetBins);
      for (int i = 0; i < targetBins; i++) {
        double melIndex = (i / (targetBins - 1)) * (melFrame.length - 1);
        int lower = melIndex.floor();
        int upper = melIndex.ceil();
        double weight = melIndex - lower;

        // Linear interpolation between Mel bands
        double interpolated =
            (1 - weight) * melFrame[lower] + weight * melFrame[upper];

        // Take square root as per Python reference: stft_mag**0.5
        expanded[i] = math.sqrt(interpolated.abs());
      }
      return expanded;
    }).toList();

    // 2. Initialize with random Phase
    List<Float64List> phases = List.generate(numFrames, (_) {
      return Float64List.fromList(List.generate(
          targetBins, (_) => math.Random().nextDouble() * 2 * math.pi));
    });

    Float64List signal = Float64List(numFrames * hopSize + frameSize);

    // 3. Iterative Griffin-Lim Reconstruction
    for (int iter = 0; iter < iterations; iter++) {
      // Step A: Inverse STFT to get Time-Domain signal
      signal = _inverseSTFT(magnitudes, phases);

      // Step B: Forward STFT on the signal to update phases (except on final pass)
      if (iter < iterations - 1) {
        phases = _forwardSTFTPhase(signal, numFrames, targetBins);
      }
    }

    return _convertToWav(signal, sampleRate);
  }

  /// Performs the Inverse Short-Time Fourier Transform (Overlap-Add)
  Float64List _inverseSTFT(List<Float64List> mags, List<Float64List> phases) {
    final output = Float64List(mags.length * hopSize + frameSize);
    final window = _hannWindow(frameSize);
    int targetBins = frameSize ~/ 2 + 1;

    for (int f = 0; f < mags.length; f++) {
      int startOffset = f * hopSize;

      for (int n = 0; n < frameSize; n++) {
        double sample = 0.0;

        // Reconstruct signal via Sum of Sines/Cosines (IFFT)
        for (int t = 0; t < targetBins; t++) {
          double angle = (2 * math.pi * t * n) / frameSize;
          // Apply combined Magnitude and Phase
          sample += mags[f][t] * math.cos(angle + phases[f][t]);
        }

        // Normalize energy and apply windowing for smooth overlap
        output[startOffset + n] += (sample / (frameSize / 2)) * window[n];
      }
    }
    return output;
  }

  /// Extracts phase from the current estimated signal
  List<Float64List> _forwardSTFTPhase(
      Float64List signal, int numFrames, int bins) {
    return List.generate(numFrames, (f) {
      final framePhases = Float64List(bins);
      int start = f * hopSize;

      for (int t = 0; t < bins; t++) {
        double re = 0.0;
        double im = 0.0;

        for (int n = 0; n < frameSize; n++) {
          if (start + n < signal.length) {
            double angle = (2 * math.pi * t * n) / frameSize;
            double val = signal[start + n];
            re += val * math.cos(angle);
            im -= val * math.sin(angle);
          }
        }
        // Compute phase angle
        framePhases[t] = math.atan2(im, re);
      }
      return framePhases;
    });
  }

  /// Scales signal to 16-bit PCM and returns Uint8List for WAV encoding
  Uint8List _convertToWav(Float64List signal, int sampleRate) {
    double maxVal = 0.0;
    for (var sample in signal) {
      if (sample.abs() > maxVal) maxVal = sample.abs();
    }

    final Int16List pcmList = Int16List(signal.length);
    if (maxVal > 0) {
      for (int i = 0; i < signal.length; i++) {
        // Peak normalization and 16-bit scaling
        pcmList[i] = ((signal[i] / maxVal) * 32767).toInt();
      }
    }
    return Uint8List.view(pcmList.buffer);
  }

  Float64List _hannWindow(int size) => Float64List.fromList(List.generate(
      size, (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1)))));
}
