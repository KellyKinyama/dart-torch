import "package:chess/chess.dart";

void main() {
  Chess chess = new Chess();
  while (!chess.game_over) {
    print('position: ' + chess.fen);
    print(chess.ascii);
    var moves = chess.moves();
    moves.shuffle();
    var move = moves[0];
    chess.move(move);
    print('move: ' + move);
  }
}