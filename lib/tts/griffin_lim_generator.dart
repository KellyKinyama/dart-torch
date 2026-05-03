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

  int get fftBins => frameSize ~/ 2 + 1;

  Uint8List generateWav(List<Float64List> magnitudes, int sampleRate) {
    final numFrames = magnitudes.length;

    // ✅ Clamp magnitudes (stability)
    final mags = magnitudes.map((frame) {
      return Float64List.fromList(
        frame.map((v) => math.max(v, 1e-6)).toList(),
      );
    }).toList();

    // ✅ Initialize random phase
    List<Float64List> phases = List.generate(numFrames, (_) {
      return Float64List.fromList(List.generate(
          fftBins, (_) => math.Random().nextDouble() * 2 * math.pi));
    });

    Float64List signal = Float64List(numFrames * hopSize + frameSize);

    for (int iter = 0; iter < iterations; iter++) {
      // ISTFT
      signal = _istft(mags, phases);

      // Recompute phase (except final iter)
      if (iter < iterations - 1) {
        final complexSpec = _stft(signal, numFrames);
        phases = complexSpec.map((frame) {
          final phase = Float64List(fftBins);
          for (int k = 0; k < fftBins; k++) {
            phase[k] = math.atan2(frame[k].imag, frame[k].real);
          }
          return phase;
        }).toList();
      }
    }

    return _toWav(signal, sampleRate);
  }

  // ─────────────────────────────
  // ✅ Proper ISTFT (Overlap-Add)
  // ─────────────────────────────
  Float64List _istft(List<Float64List> mags, List<Float64List> phases) {
    final output = Float64List(mags.length * hopSize + frameSize);
    final window = _hann(frameSize);
    final windowSum = Float64List(output.length);

    for (int f = 0; f < mags.length; f++) {
      final start = f * hopSize;

      // Build full complex spectrum
      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      for (int k = 0; k < fftBins; k++) {
        final mag = mags[f][k];
        final phase = phases[f][k];

        real[k] = mag * math.cos(phase);
        imag[k] = mag * math.sin(phase);
      }

      // Mirror (negative frequencies)
      for (int k = 1; k < fftBins - 1; k++) {
        real[frameSize - k] = real[k];
        imag[frameSize - k] = -imag[k];
      }

      // IFFT
      final time = _ifft(real, imag);

      // Overlap-add with window normalization
      for (int n = 0; n < frameSize; n++) {
        final idx = start + n;
        final w = window[n];

        output[idx] += time[n] * w;
        windowSum[idx] += w * w;
      }
    }

    // Normalize
    for (int i = 0; i < output.length; i++) {
      if (windowSum[i] > 1e-6) {
        output[i] /= windowSum[i];
      }
    }

    return output;
  }

  // ─────────────────────────────
  // ✅ STFT
  // ─────────────────────────────
  List<List<Complex>> _stft(Float64List signal, int numFrames) {
    final window = _hann(frameSize);

    return List.generate(numFrames, (f) {
      final start = f * hopSize;

      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      for (int n = 0; n < frameSize; n++) {
        final val =
            (start + n < signal.length) ? signal[start + n] * window[n] : 0.0;

        for (int k = 0; k < frameSize; k++) {
          final angle = 2 * math.pi * k * n / frameSize;
          real[k] += val * math.cos(angle);
          imag[k] -= val * math.sin(angle);
        }
      }

      return List.generate(
        fftBins,
        (k) => Complex(real[k], imag[k]),
      );
    });
  }

  // ─────────────────────────────
  // ✅ IFFT
  // ─────────────────────────────
  Float64List _ifft(Float64List real, Float64List imag) {
    final N = real.length;
    final output = Float64List(N);

    for (int n = 0; n < N; n++) {
      double sum = 0.0;

      for (int k = 0; k < N; k++) {
        final angle = 2 * math.pi * k * n / N;
        sum += real[k] * math.cos(angle) - imag[k] * math.sin(angle);
      }

      output[n] = sum / N;
    }

    return output;
  }

  // ─────────────────────────────
  // ✅ WAV encoding
  // ─────────────────────────────
  Uint8List _toWav(Float64List signal, int sampleRate) {
    double maxVal = 0.0;
    for (var s in signal) {
      if (s.abs() > maxVal) maxVal = s.abs();
    }

    final pcm = Int16List(signal.length);
    if (maxVal > 0) {
      for (int i = 0; i < signal.length; i++) {
        pcm[i] = ((signal[i] / maxVal) * 32767).toInt();
      }
    }

    return Uint8List.view(pcm.buffer);
  }

  Float64List _hann(int size) {
    return Float64List.fromList(List.generate(
      size,
      (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))),
    ));
  }
}

// ✅ Simple complex struct
class Complex {
  final double real;
  final double imag;

  Complex(this.real, this.imag);
}
