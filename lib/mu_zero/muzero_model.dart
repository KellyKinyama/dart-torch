// file: muzero_model.dart

// import 'dart:math' as math;
import '../transformer/transformer_encoder2.dart';
import '../transformer_misc/chess_transformer.dart';
import '/nn/module.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
// import 'chess_transformer.dart'; // We'll borrow concepts from here

// A container for the output of the dynamics function (g)
class DynamicsOutput {
  final Value reward;
  final ValueVector nextState;
  DynamicsOutput(this.reward, this.nextState);
}

// A container for the output of the prediction function (f)
class PredictionOutput {
  final ValueVector policyLogits;
  final Value value;
  PredictionOutput(this.policyLogits, this.value);
}

/// A single, unified model for MuZero, handling representation, dynamics, and prediction.
class MuZeroNet extends Module {
  final int embedSize;
  // ... other parameters like numHeads, numLayers

  // --- Shared Components ---
  // We can use a Transformer-like body to process states.
  final TransformerEncoder sharedEncoder;
  final List<ValueVector> pieceEmbeddings;
  final List<ValueVector> squarePositionalEmbeddings;

  // --- Specialized "Heads" ---
  // Head for predicting policy (move probabilities)
  final Layer policyHead;
  // Head for predicting value (win/loss score)
  final Layer valueHead;
  // Head for predicting reward (e.g., +1 for checkmate, 0 otherwise)
  final Layer rewardHead;
  // A network to process state + action for the dynamics function
  final Layer dynamicsStateActionProcessor;

  MuZeroNet({
    required this.embedSize,
    /* ... */
  })  : // TODO: Initialize all embeddings and layers.
        // Your ChessTransformer constructor is a great reference.
        pieceEmbeddings = /* ... */
            List.generate(13, (index) => ValueVector.fromDoubleList([0.0])),
        squarePositionalEmbeddings = /* ... */
            List.generate(128, (index) => ValueVector.fromDoubleList([0.0])),
        sharedEncoder = /* ... */ TransformerEncoder(numLayers: 2),
        // The policyHead is similar to your old `moveHead`
        policyHead =
            Layer.fromNeurons(embedSize * NUM_SQUARES, NUM_POSSIBLE_MOVES),
        // The valueHead predicts a single value from the board state
        valueHead = Layer.fromNeurons(embedSize * NUM_SQUARES, 1),
        // The rewardHead predicts a single reward value
        rewardHead = Layer.fromNeurons(embedSize * NUM_SQUARES, 1),
        // The dynamics network needs to combine a state and an action
        dynamicsStateActionProcessor = Layer.fromNeurons(
            embedSize * NUM_SQUARES + NUM_POSSIBLE_MOVES,
            embedSize * NUM_SQUARES) {
    // ...
  }

  /// Representation Function (h): Board State -> Hidden State
  ValueVector representation(List<int> boardState) {
    // This is very similar to the start of your ChessTransformer's forward method.
    // 1. Convert boardState to embeddings
    final List<ValueVector> embeddedSquares = List.generate(NUM_SQUARES, (i) {
      final pieceId = boardState[i];
      return pieceEmbeddings[pieceId] + squarePositionalEmbeddings[i];
    });

    // 2. Pass through the shared encoder to get the hidden state
    final encodedBoardFeatures =
        sharedEncoder.forwardEmbeddings(embeddedSquares);

    // 3. Flatten to get the final hidden state vector
    return ValueVector(
        encodedBoardFeatures.expand((vec) => vec.values).toList());
  }

  /// Prediction Function (f): Hidden State -> Policy & Value
  PredictionOutput prediction(ValueVector hiddenState) {
    // The hiddenState is the output from the representation or dynamics function.
    // Use the specialized heads to make predictions from this state.
    final policyLogits = policyHead.forward(hiddenState);
    final value =
        valueHead.forward(hiddenState).values.first; // Get single value

    return PredictionOutput(ValueVector(policyLogits.values), value);
  }

  /// Dynamics Function (g): Hidden State + Action -> Next State & Reward
  DynamicsOutput dynamics(ValueVector hiddenState, int actionIndex) {
    // 1. One-hot encode the action.
    final actionVector = ValueVector.fromDoubleList(
        List.generate(NUM_POSSIBLE_MOVES, (i) => i == actionIndex ? 1.0 : 0.0));

    // 2. Combine the current hidden state and the action vector.
    final combinedInput =
        ValueVector([...hiddenState.values, ...actionVector.values]);

    // 3. Process the combined input to get the next hidden state.
    // This is a simplified approach; more advanced models might use attention here.
    final nextState =
        ValueVector(dynamicsStateActionProcessor.forward(combinedInput).values);

    // 4. Predict the immediate reward from the *next* state.
    final reward = rewardHead.forward(nextState).values.first;

    return DynamicsOutput(reward, nextState);
  }

  @override
  List<Value> parameters() {
    // TODO: Return all parameters from all embeddings and layers
    return [
      ...pieceEmbeddings.expand((vec) => vec.values),
      ...squarePositionalEmbeddings.expand((vec) => vec.values),
      ...sharedEncoder.parameters(),
      ...policyHead.parameters(),
      ...valueHead.parameters(),
      ...rewardHead.parameters(),
      ...dynamicsStateActionProcessor.parameters()
    ];
  }
}
