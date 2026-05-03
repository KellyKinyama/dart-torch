import 'dart:math' as math;
import 'dart:typed_data';

class GriffinLimGenerator {
  final int iterations;
  final int frameSize;
  final int hopSize;

  GriffinLimGenerator({
    this.iterations = 80, // ✅ more iterations = better quality
    this.frameSize = 1024,
    this.hopSize = 256,
  });

  int get bins => frameSize ~/ 2 + 1;

  Uint8List generateWav(List<Float64List> magnitudes, int sampleRate) {
    final numFrames = magnitudes.length;

    // ✅ clamp magnitudes
    final mags = magnitudes
        .map((f) =>
            Float64List.fromList(f.map((v) => math.max(v, 1e-6)).toList()))
        .toList();

    // ✅ random phase
    List<Float64List> phases = List.generate(numFrames, (_) {
      return Float64List.fromList(
          List.generate(bins, (_) => math.Random().nextDouble() * 2 * math.pi));
    });

    Float64List signal = Float64List(numFrames * hopSize + frameSize);

    for (int iter = 0; iter < iterations; iter++) {
      signal = _istft(mags, phases);

      if (iter < iterations - 1) {
        phases = _estimatePhase(signal, numFrames);
      }
    }

    return _toWav(signal, sampleRate);
  }

  // ✅ ISTFT (correct overlap-add)
  Float64List _istft(List<Float64List> mags, List<Float64List> phases) {
    final out = Float64List(mags.length * hopSize + frameSize);
    final window = _hann(frameSize);
    final norm = Float64List(out.length);

    for (int f = 0; f < mags.length; f++) {
      final start = f * hopSize;

      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      // ✅ positive freqs
      for (int k = 0; k < bins; k++) {
        real[k] = mags[f][k] * math.cos(phases[f][k]);
        imag[k] = mags[f][k] * math.sin(phases[f][k]);
      }

      // ✅ mirror (skip DC & Nyquist)
      for (int k = 1; k < bins - 1; k++) {
        real[frameSize - k] = real[k];
        imag[frameSize - k] = -imag[k];
      }

      final frame = _ifft(real, imag);

      for (int n = 0; n < frameSize; n++) {
        final idx = start + n;
        final w = window[n];

        out[idx] += frame[n] * w;
        norm[idx] += w * w;
      }
    }

    for (int i = 0; i < out.length; i++) {
      if (norm[i] > 1e-6) {
        out[i] /= norm[i];
      }
    }

    return out;
  }

  // ✅ Phase estimation (STFT projection)
  List<Float64List> _estimatePhase(Float64List signal, int numFrames) {
    final window = _hann(frameSize);

    return List.generate(numFrames, (f) {
      final start = f * hopSize;

      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      for (int n = 0; n < frameSize; n++) {
        double x =
            (start + n < signal.length) ? signal[start + n] * window[n] : 0.0;

        for (int k = 0; k < bins; k++) {
          double angle = 2 * math.pi * k * n / frameSize;
          real[k] += x * math.cos(angle);
          imag[k] -= x * math.sin(angle);
        }
      }

      final phase = Float64List(bins);
      for (int k = 0; k < bins; k++) {
        phase[k] = math.atan2(imag[k], real[k]);
      }

      return phase;
    });
  }

  // ✅ IFFT (correct scaling)
  Float64List _ifft(Float64List real, Float64List imag) {
    final N = real.length;
    final out = Float64List(N);

    for (int n = 0; n < N; n++) {
      double sum = 0.0;

      for (int k = 0; k < N; k++) {
        double angle = 2 * math.pi * k * n / N;
        sum += real[k] * math.cos(angle) - imag[k] * math.sin(angle);
      }

      out[n] = sum / N;
    }

    return out;
  }

  Uint8List _toWav(Float64List signal, int sr) {
    double maxVal = 0.0;
    for (var s in signal) {
      if (s.abs() > maxVal) maxVal = s.abs();
    }

    final pcm = Int16List(signal.length);

    for (int i = 0; i < signal.length; i++) {
      pcm[i] = ((signal[i] / (maxVal + 1e-9)) * 32767).toInt();
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
