// file: chess_transformer.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '../transformer2/transformer_encoder2.dart'; // Your TransformerEncoder

// --- Chess-specific constants (Simplified) ---
// These define how pieces are mapped to integers and board dimensions.
// In a real system, you might use FEN strings or more detailed board state objects.
const int EMPTY_SQUARE_ID = 0;
const int WHITE_PAWN = 1;
const int WHITE_KNIGHT = 2;
const int WHITE_BISHOP = 3;
const int WHITE_ROOK = 4;
const int WHITE_QUEEN = 5;
const int WHITE_KING = 6;
const int BLACK_PAWN = 7;
const int BLACK_KNIGHT = 8;
const int BLACK_BISHOP = 9;
const int BLACK_ROOK = 10;
const int BLACK_QUEEN = 11;
const int BLACK_KING = 12;
const int NUM_PIECE_TYPES = 13; // Includes 0 for empty

const int BOARD_SIZE = 8; // 8x8 chess board
const int NUM_SQUARES = BOARD_SIZE * BOARD_SIZE; // 64 squares

// Simplified move representation: (from_square_idx * NUM_SQUARES) + to_square_idx
// This results in 64*64 = 4096 possible (start, end) pairs, many of which are illegal.
// A real system would filter legal moves or use a more nuanced action space.
const int NUM_POSSIBLE_MOVES = NUM_SQUARES * NUM_SQUARES;

/// A Transformer model adapted for Chess board state encoding and move prediction.
///
/// This model takes a flattened representation of the chess board (piece IDs
/// on each square) and predicts logits for possible moves.
class ChessTransformer extends Module {
  final int embedSize; // Transformer embedding dimension
  final int numLayers; // Number of encoder layers
  final int numHeads; // Number of attention heads

  // Learnable embeddings for each possible piece type (including empty)
  final List<ValueVector> pieceEmbeddings;

  // Learnable positional embeddings for each of the 64 squares
  // These help the Transformer know "where" on the board a piece is.
  final List<ValueVector> squarePositionalEmbeddings;

  // The main Transformer Encoder backbone
  final TransformerEncoder transformerEncoder;

  // Final head for predicting move logits
  // It takes the flattened output of the Transformer Encoder (64 * embedSize)
  // and projects it to the total number of possible (start, end) moves.
  final Layer moveHead;

  ChessTransformer({
    required this.embedSize,
    this.numLayers = 2, // Reduced for faster example execution
    this.numHeads = 4, // Reduced for faster example execution
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        // Initialize piece embeddings
        pieceEmbeddings = List.generate(
            NUM_PIECE_TYPES,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // Initialize square positional embeddings
        squarePositionalEmbeddings = List.generate(
            NUM_SQUARES,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // The TransformerEncoder processes the sequence of 64 square embeddings
        transformerEncoder = TransformerEncoder(
          vocabSize: 0, // Not used, as embeddings are provided directly
          embedSize: embedSize,
          blockSize: NUM_SQUARES, // Sequence length is 64 (for 64 squares)
          numLayers: numLayers,
          numHeads: numHeads,
        ),
        // The moveHead takes a flattened representation of the entire board's
        // encoded features (64 * embedSize) and outputs logits for all possible moves.
        moveHead =
            Layer.fromNeurons(NUM_SQUARES * embedSize, NUM_POSSIBLE_MOVES);

  /// The forward pass for the Chess Transformer.
  ///
  /// Takes a flattened list of piece IDs representing the 64 squares of the board.
  /// Returns logits for all possible (start_square, end_square) moves.
  ///
  /// [boardState]: A List<int> of length 64, where each int is a piece ID.
  List<Value> forward(List<int> boardState) {
    if (boardState.length != NUM_SQUARES) {
      throw ArgumentError(
          "Board state must contain exactly $NUM_SQUARES piece IDs.");
    }

    // 1. Convert each square's piece ID into an embedding and add positional encoding
    final List<ValueVector> embeddedSquares = List.generate(NUM_SQUARES, (i) {
      final pieceId = boardState[i];
      if (pieceId < 0 || pieceId >= NUM_PIECE_TYPES) {
        throw ArgumentError("Invalid piece ID: $pieceId at square $i");
      }
      return pieceEmbeddings[pieceId] + squarePositionalEmbeddings[i];
    });

    // 2. Pass the sequence of embedded squares through the Transformer Encoder
    final encodedBoardFeatures =
        transformerEncoder.forwardEmbeddings(embeddedSquares);

    // 3. Flatten the output of the Transformer Encoder
    // The Transformer outputs a list of 64 ValueVectors (one for each square).
    // We concatenate all these vectors into a single large vector.
    final flattenedFeatures =
        ValueVector(encodedBoardFeatures.expand((vec) => vec.values).toList());

    // 4. Pass the flattened features through the move prediction head
    final moveLogits = moveHead.forward(flattenedFeatures);

    return moveLogits
        .values; // Return a list of Value objects representing logits for each move
  }

  @override
  List<Value> parameters() {
    return [
      ...pieceEmbeddings.expand((vec) => vec.values),
      ...squarePositionalEmbeddings.expand((vec) => vec.values),
      ...transformerEncoder.parameters(),
      ...moveHead.parameters(),
    ];
  }
}
