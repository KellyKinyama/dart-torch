import 'dart:io';
import 'dart:typed_data';

import '../stft_spectrogram.dart';
import 'griffin_lim_generator.dart';

Future<void> main() async {
  print('---- TESTING GRIFFIN-LIM ----');

  const sampleRate = 16000;
  const frameSize = 1024;
  const hopSize = 256;
  const maxFrames = 64;

  // 1) Load spectrogram frames
  final spectrogram = await stftSpectrogram(
    'output.wav',
    frameSize: frameSize,
    hopSize: hopSize,
  );

  // 2) Convert to Float64List frames and sanitize values
  // Griffin-Lim expects LINEAR magnitudes (not dB). [1](https://pub.dev/documentation/mp3/latest/)[2](https://github.com/JamesHeinrich/getID3)
  final magnitudes = spectrogram.take(maxFrames).map((frame) {
    final out = Float64List(frame.length);
    for (int i = 0; i < frame.length; i++) {
      final v = frame[i];
      // Keep finite, positive magnitudes
      out[i] = (v.isFinite ? v : 0.0).clamp(1e-8, 1e8);
    }
    return out;
  }).toList();

  print('Frames: ${magnitudes.length}');
  print('Bins:   ${magnitudes.isNotEmpty ? magnitudes[0].length : 0}');

  // 3) Run Griffin–Lim
  final griffin = GriffinLimGenerator(
    iterations: 80,
    frameSize: frameSize,
    hopSize: hopSize,
  );

  print('Running Griffin–Lim...');
  final bytes = griffin.generateWav(magnitudes, sampleRate);

  // 4) Ensure the output is a real WAV file (RIFF header).
  final wavBytes = _ensureWavPcm16Mono(bytes, sampleRate);

  await File('test_griffin.wav').writeAsBytes(wavBytes);

  print('\n✅ Saved: test_griffin.wav');
}

/// If [bytes] already looks like a WAV (RIFF/WAVE), return it as-is.
/// Otherwise, treat it as raw PCM16 LE mono and wrap with a WAV header.
Uint8List _ensureWavPcm16Mono(Uint8List bytes, int sampleRate) {
  if (_looksLikeWav(bytes)) return bytes;
  return _wrapRawPcm16leToWav(bytes, sampleRate, channels: 1);
}

bool _looksLikeWav(Uint8List bytes) {
  if (bytes.length < 12) return false;
  return bytes[0] == 0x52 && // R
      bytes[1] == 0x49 && // I
      bytes[2] == 0x46 && // F
      bytes[3] == 0x46 && // F
      bytes[8] == 0x57 && // W
      bytes[9] == 0x41 && // A
      bytes[10] == 0x56 && // V
      bytes[11] == 0x45; // E
}

/// Wrap raw PCM16 little-endian bytes with a standard WAV header.
Uint8List _wrapRawPcm16leToWav(
  Uint8List pcmBytes,
  int sampleRate, {
  int channels = 1,
}) {
  const bitsPerSample = 16;
  final bytesPerSample = bitsPerSample ~/ 8;
  final blockAlign = channels * bytesPerSample;
  final byteRate = sampleRate * blockAlign;

  final dataSize = pcmBytes.length;
  final riffSize = 36 + dataSize;

  final header = ByteData(44);
  void writeAscii(int offset, String s) {
    for (int i = 0; i < s.length; i++) {
      header.setUint8(offset + i, s.codeUnitAt(i));
    }
  }

  writeAscii(0, 'RIFF');
  header.setUint32(4, riffSize, Endian.little);
  writeAscii(8, 'WAVE');

  writeAscii(12, 'fmt ');
  header.setUint32(16, 16, Endian.little); // fmt chunk size
  header.setUint16(20, 1, Endian.little); // PCM
  header.setUint16(22, channels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, byteRate, Endian.little);
  header.setUint16(32, blockAlign, Endian.little);
  header.setUint16(34, bitsPerSample, Endian.little);

  writeAscii(36, 'data');
  header.setUint32(40, dataSize, Endian.little);

  final out = Uint8List(44 + dataSize);
  out.setAll(0, header.buffer.asUint8List());
  out.setAll(44, pcmBytes);
  return out;
}
