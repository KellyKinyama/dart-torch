class EnglishCharacterTokenizer {
  final Map<String, int> _charToId = {};
  final Map<int, String> _idToChar = {};

  // Special Token Constants
  static const String pad = '<PAD>'; // ID 0
  static const String sos = '<SOS>'; // ID 1
  static const String eos = '<EOS>'; // ID 2
  static const String unk = '<UNK>'; // ID 3

  EnglishCharacterTokenizer() {
    // 1. Initialize special tokens
    final specialTokens = [pad, sos, eos, unk];
    for (int i = 0; i < specialTokens.length; i++) {
      _addToken(specialTokens[i]);
    }

    // 2. Add Standard English Set:
    // Lowercase (a-z)
    _addRange(97, 122);
    // Uppercase (A-Z)
    _addRange(65, 90);
    // Digits (0-9)
    _addRange(48, 57);
    // Punctuation and Space
    final punctuation = " !\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    for (var i = 0; i < punctuation.length; i++) {
      _addToken(punctuation[i]);
    }
  }

  void _addToken(String char) {
    if (!_charToId.containsKey(char)) {
      final id = _charToId.length;
      _charToId[char] = id;
      _idToChar[id] = char;
    }
  }

  void _addRange(int start, int end) {
    for (int i = start; i <= end; i++) {
      _addToken(String.fromCharCode(i));
    }
  }

  int get vocabSize => _charToId.length;

  /// Encodes string to IDs with SOS and EOS
  List<int> encode(String text, {int? maxLen}) {
    List<int> ids = [_charToId[sos]!];
    
    for (int i = 0; i < text.length; i++) {
      ids.add(_charToId[text[i]] ?? _charToId[unk]!);
    }
    
    ids.add(_charToId[eos]!);

    if (maxLen != null) {
      if (ids.length > maxLen) {
        // Truncate and ensure it ends with EOS
        ids = ids.sublist(0, maxLen - 1)..add(_charToId[eos]!);
      } else {
        // Pad with PAD token
        while (ids.length < maxLen) {
          ids.add(_charToId[pad]!);
        }
      }
    }
    return ids;
  }

  /// Decodes IDs back to string
  String decode(List<int> ids) {
    return ids
        .map((id) => _idToChar[id] ?? unk)
        .where((char) => ![pad, sos, eos].contains(char))
        .join('');
  }
}