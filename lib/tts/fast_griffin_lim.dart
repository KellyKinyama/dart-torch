import 'dart:math' as math;
import 'dart:typed_data';

class FastGriffinLimGenerator {
  final int iterations;
  final int frameSize;
  final int hopSize;
  final double momentum;

  FastGriffinLimGenerator({
    this.iterations = 60,
    this.frameSize = 1024,
    this.hopSize = 256,
    this.momentum =
        0.99, // 0.0 => classic Griffin–Lim [1](https://pub.dev/documentation/mp3/latest/)
  });

  int get bins => frameSize ~/ 2 + 1;

  Uint8List generateWav(List<Float64List> magnitudes, int sampleRate) {
    final numFrames = magnitudes.length;

    // Clamp magnitudes (avoid zeros)
    final mags = magnitudes.map((f) {
      final out = Float64List(bins);
      for (int k = 0; k < bins; k++) {
        final v = f[k];
        out[k] = (v.isFinite ? v : 0.0).clamp(1e-8, 1e8);
      }
      return out;
    }).toList();

    // Random initial phase
    final rng = math.Random(12345);
    var phases = List.generate(numFrames, (_) {
      final p = Float64List(bins);
      for (int k = 0; k < bins; k++) {
        p[k] = rng.nextDouble() * 2.0 * math.pi;
      }
      return p;
    });

    // For fast variant: keep previous complex estimate X_{k-1}
    List<Float64List>? prevReal;
    List<Float64List>? prevImag;

    Float64List signal = Float64List(numFrames * hopSize + frameSize);

    for (int iter = 0; iter < iterations; iter++) {
      // ISTFT using current phases
      signal = _istft(mags, phases);

      // Estimate phase (STFT projection)
      final estPhase = _estimatePhase(signal, numFrames);

      // Build complex spectrum from mags + estimated phase
      final curReal = List.generate(numFrames, (_) => Float64List(bins));
      final curImag = List.generate(numFrames, (_) => Float64List(bins));

      for (int t = 0; t < numFrames; t++) {
        for (int k = 0; k < bins; k++) {
          final ph = estPhase[t][k];
          final mag = mags[t][k];
          curReal[t][k] = mag * math.cos(ph);
          curImag[t][k] = mag * math.sin(ph);
        }
      }

      // Fast Griffin–Lim acceleration (momentum)
      // Z = X + mu*(X - Xprev), then phases = angle(Z)
      if (prevReal != null && prevImag != null && momentum > 0.0) {
        for (int t = 0; t < numFrames; t++) {
          for (int k = 0; k < bins; k++) {
            final rx = curReal[t][k];
            final ix = curImag[t][k];
            final rPrev = prevReal[t][k];
            final iPrev = prevImag[t][k];

            final rZ = rx + momentum * (rx - rPrev);
            final iZ = ix + momentum * (ix - iPrev);

            phases[t][k] = math.atan2(iZ, rZ);
          }
        }
      } else {
        // First iteration: just use estimated phase
        phases = estPhase;
      }

      prevReal = curReal;
      prevImag = curImag;
    }

    return _toRawPcm16(signal);
  }

  // ISTFT (overlap-add + window^2 normalization)
  Float64List _istft(List<Float64List> mags, List<Float64List> phases) {
    final out = Float64List(mags.length * hopSize + frameSize);
    final window = _hann(frameSize);
    final norm = Float64List(out.length);

    for (int f = 0; f < mags.length; f++) {
      final start = f * hopSize;

      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      // positive freqs
      for (int k = 0; k < bins; k++) {
        real[k] = mags[f][k] * math.cos(phases[f][k]);
        imag[k] = mags[f][k] * math.sin(phases[f][k]);
      }

      // mirror (skip DC & Nyquist)
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
      if (norm[i] > 1e-6) out[i] /= norm[i];
    }
    return out;
  }

  // Phase estimation (slow DFT, but known-working with your pipeline)
  List<Float64List> _estimatePhase(Float64List signal, int numFrames) {
    final window = _hann(frameSize);

    return List.generate(numFrames, (f) {
      final start = f * hopSize;
      final real = Float64List(frameSize);
      final imag = Float64List(frameSize);

      for (int n = 0; n < frameSize; n++) {
        final idx = start + n;
        final x = (idx < signal.length) ? signal[idx] * window[n] : 0.0;

        for (int k = 0; k < bins; k++) {
          final angle = 2 * math.pi * k * n / frameSize;
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

  // IFFT (O(N^2) but correct; you already had this working)
  Float64List _ifft(Float64List real, Float64List imag) {
    final N = real.length;
    final out = Float64List(N);

    for (int n = 0; n < N; n++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++) {
        final angle = 2 * math.pi * k * n / N;
        sum += real[k] * math.cos(angle) - imag[k] * math.sin(angle);
      }
      out[n] = sum / N;
    }
    return out;
  }

  Uint8List _toRawPcm16(Float64List signal) {
    double maxVal = 0.0;
    for (final s in signal) {
      final a = s.abs();
      if (a > maxVal) maxVal = a;
    }
    final scale = maxVal > 1e-9 ? (32767.0 / maxVal) : 0.0;

    final pcm = Int16List(signal.length);
    for (int i = 0; i < signal.length; i++) {
      pcm[i] = (signal[i] * scale).clamp(-32768.0, 32767.0).round();
    }
    return pcm.buffer.asUint8List();
  }

  Float64List _hann(int size) {
    final w = Float64List(size);
    for (int n = 0; n < size; n++) {
      w[n] = 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1)));
    }
    return w;
  }
}
