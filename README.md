Dart Deep Learning Library Documentation
This document provides an overview and detailed explanation of a custom deep learning micro-framework implemented in Dart. This library focuses on providing core building blocks for neural networks, including automatic differentiation (Value class), basic layers (Neuron, Layer), and components for Transformer architectures. It also includes utilities for saving and loading model weights.

Table of Contents
Core Primitives

Value

ValueVector

Base Module

Module

Neural Network Layers

Neuron

Layer

Transformer Architecture Components

TransformerBlock

TransformerDecoderBlock

EncoderDecoderTransformer

Utilities

network_utils.dart

Training Considerations

Getting Started

Examples

Basic Transformer Training

Encoder-Decoder Transformer Training and Inference

GPT Training

GPT Generation

Advanced Transformer Component Examples

1. Core Primitives
These classes form the fundamental mathematical and computational backbone of the library, enabling automatic differentiation.

Value
(value.dart)

The Value class is the cornerstone of the automatic differentiation system. Each Value object holds a single scalar data (a double) and its corresponding grad (gradient), which accumulates during backpropagation. It tracks the operations performed to create it (_op) and its immediate predecessors (_prev), forming a computational graph.

Properties:

double data: The numerical value of this node in the computation graph.

double grad: The gradient of the loss with respect to this value, computed during backward(). Initialized to 0.0.

void Function() _backward: A closure that defines how to compute gradients for this specific operation. This is the crucial part of reverse-mode autodiff, specifying how the gradient of the output (out.grad) should be distributed to the grad of its direct input Value objects.

Set<Value> _prev: A set of Value objects that are direct inputs to the operation that produced this Value.

String _op: A string representing the operation that created this Value (e.g., '+', '*', 'ReLU'). This is primarily for debugging and visualization.

Constructors:

Value(this.data, [Set<Value>? children, this._op = '']): Creates a new Value instance.

static Value toValue(dynamic x): Converts a num or Value to a Value object.

Operators & Methods:
The Value class overloads standard arithmetic operators and provides common activation functions. Each overloaded operator and activation function creates a new Value object and crucially defines its _backward closure, which correctly applies the chain rule for gradient propagation.

Value operator +(dynamic other): Addition. If out = a + b, then 
fracpartialtextoutpartiala=1 and 
fracpartialtextoutpartialb=1. So, a.grad += out.grad and b.grad += out.grad.

Value operator *(dynamic other): Multiplication. If out = a * b, then 
fracpartialtextoutpartiala=b and 
fracpartialtextoutpartialb=a. So, a.grad += b.data * out.grad and b.grad += a.data * out.grad.

Value operator -(): Unary negation.

Value operator -(dynamic other): Subtraction.

Value operator /(dynamic other): Division. If out = a / b, then 
fracpartialtextoutpartiala=
frac1b and 
fracpartialtextoutpartialb=−
fracab 
2
 . So, a.grad += (1 / b.data) * out.grad and b.grad += (-a.data / (b.data * b.data)) * out.grad.

Value pow(num exponent): Power function (x 
n
 ). If out = x.pow(n), then n
cdotx 
n−1
 . So, x.grad += (exponent * math.pow(data, exponent - 1).toDouble()) * out.grad.

Value abs(): Absolute value.

Value relu(): Rectified Linear Unit activation. If x0, y=x, so 
fracpartialypartialx=1. If x
le0, y=0, so 
fracpartialypartialx=0. The _backward implementation correctly sets grad += (out.data > 0.0 ? 1.0 : 0.0) * out.grad.

Value sigmoid(): Sigmoid activation. If y=
sigma(x)=
frac11+e 
−x
 , then 
fracpartialypartialx=
sigma(x)(1−
sigma(x)). The _backward implementation uses grad += s * (1 - s) * out.grad where s is the sigmoid output.

Value tanh(): Hyperbolic Tangent activation.

Value elu(double alpha): Exponential Linear Unit activation.

Value gelu(): Gaussian Error Linear Unit activation.

Value exp(): Exponential function (e 
x
 ).

Value log(): Natural logarithm.

Value sqrt(): Square root.

static List<Value> softmax(List<Value> inputs): Computes softmax for a list of Values. The gradients for softmax are more complex as each output depends on all inputs. The _backward method for each Value produced by softmax internally handles the gradient propagation.

void zeroGrad(): Recursively sets grad to 0.0 for this Value and all its ancestors in the computation graph. This is essential before each new backward pass to ensure gradients from previous computations don't interfere.

void backward(): Initiates the backpropagation process. It first constructs a topological sort of the computational graph starting from the Value on which backward() is called, ensuring all children's gradients are computed before propagating to their parents. The grad of the output Value is initialized to 1.0 (as 
fracpartialtextoutputpartialtextoutput=1). It then iterates through the topologically sorted list in reverse order, calling v._backward() for each Value v, which distributes v.grad to its direct inputs.

void setData(double newData): Sets a new data value for the Value.

ValueVector
(value_vector.dart)

A ValueVector is a wrapper around a List<Value>, providing vector-like operations. It's used to represent vectors of parameters (like weights in a neuron) or inputs/outputs from layers.

Properties:

final List<Value> values: The list of Value objects composing the vector.

Constructors:

ValueVector(this.values): Creates a ValueVector from a list of Values.

ValueVector.fromFloat32List(Float32List data): Creates from a Float32List.

ValueVector.fromUint8List(Uint8List data): Creates from a Uint8List.

ValueVector.fromDoubleList(List<double> data): Creates from a List<double>.

Methods & Operators:

Value dot(ValueVector other): Computes the dot product with another ValueVector.

ValueVector operator +(dynamic other): Element-wise addition with a Value scalar or another ValueVector.

ValueVector operator /(Value other): Element-wise division by a Value scalar.

ValueVector operator *(Value other): Element-wise multiplication by a Value scalar.

ValueVector operator -(ValueVector other): Element-wise subtraction with another ValueVector.

ValueVector squared(): Element-wise square.

Value mean(): Computes the mean of all Values in the vector.

Value crossEntropy(ValueVector target): Computes cross-entropy loss.

ValueVector sigmoid(): Applies sigmoid activation element-wise.

ValueVector softmax(): Applies softmax activation to the vector.

ValueVector reLU(): Applies ReLU activation element-wise.

2. Base Module
Module
(module.dart)

The Module class serves as the abstract base class for all trainable components of the neural network. It defines a common interface for managing parameters and gradients.

Methods:

void zeroGrad(): Sets the grad property of all parameters within the module (and its sub-modules) to 0.0.

List<Value> parameters(): Abstract method. Subclasses must implement this to return a flat list of all Value objects that constitute the trainable parameters of the module. It is crucial that this method only returns learnable parameters (weights and biases) and not intermediate activation Value objects.

3. Neural Network Layers
These classes implement basic building blocks of a neural network.

Neuron
(neuron.dart)

A Neuron represents a single computational unit in a neural network, performing a weighted sum of its inputs and optionally applying a non-linear activation.

Properties:

ValueVector w: The weights of the neuron, represented as a ValueVector.

Value? b: The bias of the neuron, represented as a single Value. Can be null if no bias is used.

bool nonlin: A flag indicating whether a non-linear activation function should be applied.

Constructors:

Neuron(this.w, {this.b, this.nonlin = true}): Creates a neuron with specified weights and bias.

factory Neuron.fromWeights(int nin, {bool nonlin = true}): A factory constructor to create a neuron with nin input connections. It initializes weights using He initialization, which is suitable for ReLU activations, by sampling from a distribution with a standard deviation of 
sqrt2/textnin. Biases are initialized to 0.0.

Methods:

Value forward(ValueVector x): Computes the output of the neuron given an input ValueVector x. It performs a dot product of weights w and input x, then adds the bias b if present. If nonlin is true, it applies the ReLU activation function to the output.

@override List<Value> parameters(): Returns a list containing all the Value objects that are part of this neuron's weights (w) and bias (b). The bias is only included if it is not null.

Layer
(layer.dart)

A Layer is a collection of Neurons, typically forming a single hidden or output layer in a multi-layer perceptron.

Properties:

List<Neuron> neurons: A list of Neuron objects that make up this layer.

Constructors:

Layer(this.neurons): Creates a layer from an existing list of neurons.

factory Layer.fromNeurons(int nin, int nout): A factory constructor to create a layer with nout neurons, each expecting nin inputs. It uses Neuron.fromWeights for neuron initialization.

Methods:

ValueVector forward(ValueVector x): Performs a forward pass through all neurons in the layer. It takes an input ValueVector x and returns a ValueVector where each element is the output of a corresponding neuron in the layer.

@override List<Value> parameters(): Collects and returns all trainable parameters (Value objects) from every neuron within this layer.

4. Transformer Architecture Components
These classes are designed to build a Transformer model for sequence-to-sequence tasks. They assume the existence of standard components like MultiHeadAttention, FeedForward, and LayerNorm.

TransformerBlock
(transformer_block.dart)

A TransformerBlock represents a single block within a Transformer Encoder. It consists of a Multi-Head Attention mechanism followed by a Feed-Forward Network, with residual connections and layer normalization applied before each sub-layer.

Properties:

final MultiHeadAttention attention: The multi-head self-attention sub-layer.

final FeedForward ffn: The position-wise feed-forward network sub-layer.

final LayerNorm ln1: Layer normalization applied before the attention sub-layer.

final LayerNorm ln2: Layer normalization applied before the feed-forward sub-layer.

final int embedSize: The dimensionality of the embedding space.

Constructor:

TransformerBlock(this.embedSize, int numHeads, {bool masked = false}): Initializes the block with embedding size, number of attention heads, and an optional masked flag (relevant for decoder blocks).

Methods:

List<ValueVector> forward(List<ValueVector> x): Performs the forward pass through the block. It applies layer normalization, then multi-head attention, followed by another layer normalization and a feed-forward network, with residual connections after each.

@override List<Value> parameters(): Returns all trainable parameters from its sub-components (attention, ffn, ln1, ln2).

TransformerDecoderBlock
(transformer_decoder_block.dart)

A TransformerDecoderBlock is a single block within a Transformer Decoder. It extends the TransformerBlock by adding a cross-attention mechanism that attends to the encoder's output. It includes masked self-attention, cross-attention, and a feed-forward network, each with residual connections and layer normalization.

Properties:

final MultiHeadAttention selfAttention: The masked multi-head self-attention sub-layer.

final MultiHeadCrossAttention crossAttention: The multi-head cross-attention sub-layer, attending to encoder outputs.

final FeedForward ffn: The position-wise feed-forward network sub-layer.

final LayerNorm ln1: Layer normalization before self-attention.

final LayerNorm ln2: Layer normalization before cross-attention.

final LayerNorm ln3: Layer normalization before the feed-forward sub-layer.

final int embedSize: The dimensionality of the embedding space for the decoder.

Constructor:

TransformerDecoderBlock(this.embedSize, int numHeads, int encoderEmbedSize): Initializes the decoder block, requiring embedSize, numHeads, and encoderEmbedSize (for cross-attention).

Methods:

List<ValueVector> forward(List<ValueVector> x_decoder, List<ValueVector> x_encoder): Performs the forward pass. It first applies masked self-attention to x_decoder, then cross-attention using x_decoder as query and x_encoder as key/value, and finally passes through a feed-forward network. Residual connections and layer normalization are applied at each step.

@override List<Value> parameters(): Returns all trainable parameters from its sub-components (selfAttention, crossAttention, ffn, ln1, ln2, ln3).

EncoderDecoderTransformer
(encoder_decoder_transformer.dart)

The EncoderDecoderTransformer is the full sequence-to-sequence Transformer model, combining a TransformerEncoder and a TransformerDecoder.

Properties:

final TransformerEncoder encoder: The encoder part of the Transformer.

final TransformerDecoder decoder: The decoder part of the Transformer.

Constructor:

EncoderDecoderTransformer({required int sourceVocabSize, required int targetVocabSize, required int embedSize, required int sourceBlockSize, required int targetBlockSize, required int numLayers, required int numHeads}): Initializes the encoder and decoder components with specified vocabulary sizes, embedding size, block sizes, number of layers, and number of attention heads.

Methods:

List<ValueVector> forward(List<int> sourceIdx, List<int> targetIdx): Performs a full forward pass through the Encoder-Decoder Transformer. It first encodes the sourceIdx using the encoder, then decodes the targetIdx using the decoder, leveraging the encoder's output via cross-attention.

@override List<Value> parameters(): Returns all trainable parameters from both the encoder and decoder components.

Note on Assumed Components:
The Transformer components (TransformerBlock, TransformerDecoderBlock, EncoderDecoderTransformer) rely on several other modules that were not explicitly provided in the analysis but are standard in Transformer architectures. These include:

MultiHeadAttention

MultiHeadCrossAttention

FeedForward

LayerNorm

TransformerEncoder

TransformerDecoder

It is assumed that these modules are implemented elsewhere in the library, correctly extending Module and providing their respective parameters() methods.

5. Utilities
network_utils.dart
This file provides utility functions for persisting and loading the trainable parameters of Module instances, typically for saving and restoring trained neural network models.

Functions:

Future<void> saveModuleParameters(Module module, String filePath)

Purpose: Saves the current data values of all trainable parameters from a given Module to a JSON file.

Parameters:

module: The Module instance (e.g., your EncoderDecoderTransformer or a Layer) whose parameters you want to save.

filePath: The full path to the JSON file where the parameters will be stored.

Serialization Format: The parameters are extracted as a List<double> (the data field of each Value object) and then encoded as a JSON array.

Usage:

import 'network_utils.dart';
// ... create your model ...
await saveModuleParameters(myModel, 'my_model_weights.json');

Future<void> loadModuleParameters(Module module, String filePath)

Purpose: Loads parameters (weights) from a JSON file and updates the data values of the corresponding Value objects within a given Module.

Parameters:

module: The Module instance into which the loaded parameters will be set. This module should have the same architecture and number of parameters as the one from which the weights were saved.

filePath: The full path to the JSON file containing the saved weights.

Important Note: This function relies heavily on the order of parameters returned by module.parameters() matching the order in which they were saved. Any change in the model's architecture or the parameters() implementation can lead to incorrect weight loading.

Error Handling: Includes checks for file existence and attempts to handle FormatException if the JSON is not a list of numbers. It also prints a warning if the number of loaded parameters does not match the module's expected parameter count.

Usage:

import 'network_utils.dart';
// ... create a new model instance with the same architecture ...
await loadModuleParameters(newModel, 'my_model_weights.json');

6. Training Considerations
When training deep neural networks, especially those employing ReLU activation functions, certain practices are crucial for stable and effective learning.

Dying ReLUs
"Dying ReLUs" is a common problem where ReLU neurons become inactive during training and stop learning. This occurs when the input to a ReLU neuron consistently falls into the negative range, causing its output to be 0.0 and, consequently, its gradient to also be 0.0. If a neuron's gradient is zero, its associated weights and biases will not be updated during optimization, leading to stagnant learning and a constant loss value.

Causes of Dying ReLUs:

Poor Initialization: If weights are initialized such that many ReLU inputs are initially negative, neurons can "die" from the start of training.

Large Learning Rates: An excessively high learning rate can cause large weight updates, pushing neuron activations into the negative region.

Unnormalized Input Data: Large input values can lead to extreme pre-activation values in ReLU layers, making it more likely for neurons to fall into the negative, non-learning region.

Solutions and Best Practices:

He Initialization: For layers using ReLU activations, use initialization schemes like He initialization (as implemented in Neuron.fromWeights). This method aims to keep the variance of activations consistent across layers, preventing signals from vanishing or exploding and reducing the likelihood of neurons dying.

Input Normalization: Normalize your input data to a more manageable range (e.g., 0.0-1.0 for pixel values, or zero mean and unit variance). This helps prevent extreme input values from pushing activations into undesirable ranges and contributes to more stable and well-behaved gradients. For Uint8List pixel data (0-255), dividing by 255.0 when converting to Value objects is a common and effective normalization strategy.

Appropriate Loss Function: For multi-class classification problems with Softmax output, Cross-Entropy Loss is generally more suitable and leads to faster, more stable training compared to Mean Squared Error.

Monitor Training: Observe the loss carefully. If it remains constant or fluctuates erratically, it's a strong indicator of training issues like dying ReLUs or an inappropriate learning rate.

Alternative Activations: If dying ReLUs persist, consider using activation functions that allow a small gradient to flow even when the input is negative, such as:

Leaky ReLU: max(alpha * x, x) where alpha is a small positive constant.

Parametric ReLU (PReLU): Similar to Leaky ReLU, but alpha is a learnable parameter.

Exponential Linear Units (ELU): For x\<0, ELU outputs 
alpha
cdot(
exp(x)−1), providing a non-zero gradient.

GELU: A smoother approximation of ReLU often used in Transformers, also providing non-zero gradients for negative inputs.

7. Getting Started
To use this library, you'll typically follow these steps:

Define your network architecture: Combine Neuron, Layer, and Transformer components (TransformerBlock, TransformerDecoderBlock, EncoderDecoderTransformer) to build your desired neural network.

Initialize your model: Create an instance of your top-level Module (e.g., EncoderDecoderTransformer). Its parameters will be randomly initialized.

Perform forward passes: Use the forward() methods of your modules to compute outputs.

Compute loss: Define a loss function using Value operations. For classification, consider using crossEntropy.

Backpropagate: Call loss.backward() to compute gradients for all parameters.

Update parameters: Implement an optimization step (e.g., gradient descent) to update Value.data based on Value.grad.

Save/Load weights: Use saveModuleParameters to save trained weights and loadModuleParameters to restore them.

Example of Model Creation, Saving, and Loading:

import 'encoder_decoder_transformer.dart';
import 'network_utils.dart';
import 'value.dart';
import 'value_vector.dart';

void main() async {
  // 1. Create a Transformer model
  final transformer = EncoderDecoderTransformer(
    sourceVocabSize: 100,
    targetVocabSize: 100,
    embedSize: 32,
    sourceBlockSize: 10,
    targetBlockSize: 10,
    numLayers: 2,
    numHeads: 4,
  );

  print('Initial parameters (first 5):');
  transformer.parameters().take(5).forEach((p) => print(p.data));

  final String filePath = 'my_transformer_model.json';

  // 2. Save the model's weights
  await saveModuleParameters(transformer, filePath);

  // 3. Create a new model instance (e.g., for inference or continued training)
  final loadedTransformer = EncoderDecoderTransformer(
    sourceVocabSize: 100,
    targetVocabSize: 100,
    embedSize: 32,
    sourceBlockSize: 10,
    targetBlockSize: 10,
    numLayers: 2,
    numHeads: 4,
  );

  print('\nParameters of new model before loading (first 5):');
  loadedTransformer.parameters().take(5).forEach((p) => print(p.data));

  // 4. Load the previously saved weights into the new model
  await loadModuleParameters(loadedTransformer, filePath);

  print('\nParameters of new model after loading (first 5):');
  loadedTransformer.parameters().take(5).forEach((p) => print(p.data));

  // Now, 'loadedTransformer' has the same weights as 'transformer' had when saved.
}

8. Examples
This section provides various examples demonstrating the usage of the deep learning library for different tasks and model architectures.

Basic Transformer Training
(example.dart, example2.dart)

These examples demonstrate the fundamental training loop for a simple Transformer model. They cover model and optimizer setup, sample data generation, forward pass, loss calculation (using cross-entropy), backward pass, and parameter updates.

Key Concepts Demonstrated:

Transformer model instantiation.

SGD optimizer usage.

Next-token prediction task.

One-hot encoding for targets.

Cross-entropy loss calculation with softmax().

Gradient clearing (zeroGrad()) and parameter updates (step()).

Basic inference after training.

// file: example.dart

import 'transformer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A simple Stochastic Gradient Descent (SGD) optimizer.
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  /// Updates each parameter using its calculated gradient.
  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("🚀 Starting Transformer Training Example...");

  // 1. --- Model & Optimizer Setup ---
  final vocabSize = 10;
  final embedSize = 16;
  final blockSize = 4; // Context length

  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 2,
  );

  final optimizer = SGD(model.parameters(), 0.1);

  // 2. --- Sample Data ---
  // The model will learn to predict the next token in the sequence.
  // For input `[1, 2, 3]`, the target is `[2, 3, 4]`.
  final sampleInputs = [1, 2, 3, 4];
  final sampleTargets = [2, 3, 4, 0]; // The next token for each position

  // 3. --- Training Loop ---
  final epochs = 50;
  print("\nTraining for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // --- Forward Pass ---
    // Get the model's predictions (logits) for each position in the input sequence.
    final logits = model.forward(sampleInputs);

    // --- Loss Calculation ---
    // We use cross-entropy loss, which is standard for classification.
    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t];
      final targetAtT = sampleTargets[t];

      // Convert the integer target to a one-hot vector representation.
      final targetVector = ValueVector(List.generate(
        vocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));

      // The `crossEntropy` function expects probabilities, so we apply softmax first.
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    // Average the loss over the sequence length.
    final meanLoss = totalLoss / Value(logits.length.toDouble());

    // --- Backward Pass & Optimization ---

    // Clear old gradients before the backward pass.
    model.zeroGrad();

    // Compute gradients for all parameters starting from the loss.
    meanLoss.backward();

    // Update the model's weights using the computed gradients.
    optimizer.step();

    if (epoch % 5 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }

  print("\n✅ Training complete.");

  // 4. --- Inference Example ---
  print("\nRunning inference with a new sequence...");
  final testInputs = [1, 2, 3];
  final finalLogits = model.forward(testInputs);

  // Get the prediction for the very last token
  final lastTokenLogits = finalLogits.last.softmax();

  // Find the token with the highest probability (argmax)
  double maxProb = -1.0;
  int predictedIndex = -1;
  for (int i = 0; i < lastTokenLogits.values.length; i++) {
    if (lastTokenLogits.values[i].data > maxProb) {
      maxProb = lastTokenLogits.values[i].data;
      predictedIndex = i;
    }
  }

  print("Input: $testInputs");
  print(
      "Predicted next token: $predictedIndex (Probability: ${(maxProb * 100).toStringAsFixed(2)}%)");
}

// file: example2.dart

import 'transformer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A simple Stochastic Gradient Descent (SGD) optimizer.
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  /// Updates each parameter using its calculated gradient.
  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("🚀 Starting Transformer Training Example...");

  // 1. --- Model & Optimizer Setup ---
  final vocabSize = 10;
  final embedSize = 16;
  final blockSize = 4; // Context length

  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 2,
  );

  final optimizer = SGD(model.parameters(), 0.1);

  // 2. --- Sample Data ---
  // The model will learn to predict the next token in the sequence.
  // For input `[1, 2, 3]`, the target is `[2, 3, 4]`.
  final sampleInputs = [2, 3, 4, 5];
  final sampleTargets = [6, 4, 5, 1]; // The next token for each position

  // 3. --- Training Loop ---
  final epochs = 50;
  print("\nTraining for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // --- Forward Pass ---
    // Get the model's predictions (logits) for each position in the input sequence.
    final logits = model.forward(sampleInputs);

    // --- Loss Calculation ---
    // We use cross-entropy loss, which is standard for classification.
    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t];
      final targetAtT = sampleTargets[t];

      // Convert the integer target to a one-hot vector representation.
      final targetVector = ValueVector(List.generate(
        vocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));

      // The `crossEntropy` function expects probabilities, so we apply softmax first.
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    // Average the loss over the sequence length.
    final meanLoss = totalLoss / Value(logits.length.toDouble());

    // --- Backward Pass & Optimization ---

    // Clear old gradients before the backward pass.
    model.zeroGrad();

    // Compute gradients for all parameters starting from the loss.
    meanLoss.backward();

    // Update the model's weights using the computed gradients.
    optimizer.step();

    if (epoch % 5 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }

  print("\n✅ Training complete.");

  // 4. --- Inference Example ---
  print("\nRunning inference with a new sequence...");
  final testInputs = [2, 3, 4];
  final finalLogits = model.forward(testInputs);

  // Get the prediction for the very last token
  final lastTokenLogits = finalLogits.last.softmax();

  // Find the token with the highest probability (argmax)
  double maxProb = -1.0;
  int predictedIndex = -1;
  for (int i = 0; i < lastTokenLogits.values.length; i++) {
    if (lastTokenLogits.values[i].data > maxProb) {
      maxProb = lastTokenLogits.values[i].data;
      predictedIndex = i;
    }
  }

  print("Input: $testInputs");
  print(
      "Predicted next token: $predictedIndex (Probability: ${(maxProb * 100).toStringAsFixed(2)}%)");
}

Encoder-Decoder Transformer Training and Inference
(example_encoder_decoder.dart)

This example demonstrates the training and a simplified greedy inference process for an EncoderDecoderTransformer model, suitable for sequence-to-sequence tasks like machine translation.

Key Concepts Demonstrated:

EncoderDecoderTransformer setup with source and target vocabulary sizes.

Teacher forcing for decoder input during training.

Cross-entropy loss calculation for sequence prediction.

Greedy decoding for inference, where the most probable token is chosen at each step.

Handling of special tokens (e.g., START_TOKEN).

// file: main.dart (or example_encoder_decoder.dart)

import 'encoder_decoder_transformer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

// Re-using SGD from previous examples
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- Encoder-Decoder Transformer Example ---");

  final sourceVocabSize = 10; // e.g., English words
  final targetVocabSize = 10; // e.g., French words
  final embedSize = 32;
  final sourceBlockSize = 8;
  final targetBlockSize = 8;
  final numLayers = 2;
  final numHeads = 4;

  // Initialize the Encoder-Decoder Transformer
  final model = EncoderDecoderTransformer(
    sourceVocabSize: sourceVocabSize,
    targetVocabSize: targetVocabSize,
    embedSize: embedSize,
    sourceBlockSize: sourceBlockSize,
    targetBlockSize: targetBlockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  final optimizer = SGD(model.parameters(), 0.05);

  // --- Sample Data for a simple sequence-to-sequence task ---
  // E.g., translating [1, 2, 3] to [5, 6, 7]
  // In real life, you'd use padding tokens and special start/end tokens.

  // Source sequence (e.g., "The dog barks")
  final sampleSourceInputs = [1, 2, 3, 4]; // Example token IDs

  // Target sequence (e.g., "Le chien aboie")
  // For training, target inputs are typically shifted right (teacher forcing).
  // If target sequence is [5, 6, 7, 8], input to decoder would be [START_TOKEN, 5, 6, 7]
  // and targets for loss would be [5, 6, 7, 8]. Let's simplify and use
  // target_inputs as the tokens given to the decoder, and target_outputs as what we want it to predict.
  final startToken = 0; // Assuming 0 is a special start-of-sequence token
  final sampleTargetInputs = [
    startToken,
    5,
    6,
    7
  ]; // Decoder input (shifted right)
  final sampleTargetOutputs = [
    5,
    6,
    7,
    8
  ]; // True next tokens for loss calculation

  if (sampleTargetInputs.length != sampleTargetOutputs.length) {
    throw ArgumentError(
        "Sample target inputs and outputs must have same length for this example.");
  }

  final epochs = 100;
  print("\nTraining Encoder-Decoder Transformer for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // Forward pass
    final logits = model.forward(sampleSourceInputs, sampleTargetInputs);

    // Calculate loss (only for the actual predicted tokens, excluding the START_TOKEN position)
    Value totalLoss = Value(0.0);
    // Iterate from 1 because targetInputs[0] is START_TOKEN, we want to predict targetOutputs[0]
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t]; // Logits for predicting targetOutputs[t]
      final targetAtT = sampleTargetOutputs[t];

      final targetVector = ValueVector(List.generate(
        targetVocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    final meanLoss = totalLoss / Value(logits.length.toDouble());

    // Backward pass & optimization
    model.zeroGrad();
    meanLoss.backward();
    optimizer.step();

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Encoder-Decoder Transformer training complete.");

  // --- Inference Example (Simplified Greedy Decoding) ---
  print("\n--- Encoder-Decoder Inference ---");
  final inferenceSource = [1, 2, 3]; // New source sequence to translate

  print("Source: $inferenceSource");
  List<int> generatedTargetSequence = [
    startToken
  ]; // Start with the start token
  final int maxGenerationLength = 5; // Max tokens to generate

  for (int i = 0; i < maxGenerationLength; i++) {
    // Encoder processes the source
    final encoderOut = model.encoder.forward(inferenceSource);

    // Decoder gets its current generated sequence as input and encoder output
    final decoderLogits =
        model.decoder.forward(generatedTargetSequence, encoderOut);

    // Get the logits for the *last* token generated by the decoder
    final lastTokenLogits = decoderLogits.last.softmax();

    // Greedy sampling: pick the token with the highest probability
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < lastTokenLogits.values.length; j++) {
      if (lastTokenLogits.values[j].data > maxProb) {
        maxProb = lastTokenLogits.values[j].data;
        predictedNextToken = j;
      }
    }

    // Add the predicted token to the generated sequence
    generatedTargetSequence.add(predictedNextToken);

    // Stop if an end-of-sequence token is predicted (you'd define one in your vocab)
    // For this example, we don't have an explicit end token, so we'll just generate `maxGenerationLength` tokens.
  }

  print(
      "Generated Target Sequence: $generatedTargetSequence (first token is START_TOKEN)");
  print(
      "Note: For real-world use, you'd handle padding, special tokens (EOS, PAD), and more advanced decoding strategies like beam search.");
}

GPT Training
(example_gpt_training.dart, example_gpt_training2.dart)

These examples showcase how to train a Generative Pretrained Transformer (GPT) model, which is essentially a TransformerDecoder used for language modeling (predicting the next token in a sequence). They include vocabulary setup, dummy dataset creation, and a training loop using cross-entropy loss.

Key Concepts Demonstrated:

Using TransformerDecoder as a GPT model.

Vocabulary mapping (stoi, itos).

Creating input and target sequences for next-token prediction.

Handling padding tokens during loss calculation.

Simplified cross-entropy loss for language modeling.

Basic greedy text generation after training.

// file: example_gpt_training.dart


// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
// Import your TransformerDecoder
import 'transformer_decoder.dart'; // Your existing TransformerDecoder
// Ensure all sub-modules of TransformerDecoder are accessible:
// transformer_decoder_block.dart, multi_head_attention.dart,
// feed_forward.dart, layer_norm2.dart, self_attention.dart,
// cross_attention.dart, multi_head_cross_attention.dart

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    
    for (final p in parameters) {
      // Only update if gradient exists
      p.data -= learningRate * p.grad;
        }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

void main() {
  print("--- Generative Pretrained Transformer (GPT) Training Example ---");

  // 1. Define GPT Model Hyperparameters
  const int vocabSize = 20; // Example vocabulary size
  const int embedSize = 32;
  const int blockSize = 10; // Maximum sequence length the GPT can process
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Simple Vocabulary for demonstration
  // This vocabulary must be consistent between training and inference
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "generation": 6,
    "model": 7,
    "the": 8,
    "quick": 9,
    "brown": 10,
    "fox": 11,
    "jumps": 12,
    "over": 13,
    "lazy": 14,
    "dog": 15,
    ".": 16, // End of sentence token
    "<start>": 17, // Start of sequence token
    "<pad>": 18, // Padding token
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Get special token IDs
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Create a Dummy Dataset
  // In a real scenario, this would be loaded from files, tokenized, and batched.
  // We'll create a few simple sequences for next-token prediction.
  // E.g., "hello world ." -> input: "<start> hello world", target: "hello world ."

  final List<List<int>> rawSequences = [
    [startTokenId, stoi["hello"]!, stoi["world"]!, stoi["."]!],
    [
      startTokenId,
      stoi["this"]!,
      stoi["is"]!,
      stoi["a"]!,
      stoi["test"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      stoi["."]!
    ],
  ];

  List<List<int>> trainInputs = [];
  List<List<int>> trainTargets = [];

  for (var seq in rawSequences) {
    // Input sequence: all tokens except the last one
    List<int> input = seq.sublist(0, seq.length - 1);
    // Target sequence: all tokens except the first one (what we predict)
    List<int> target = seq.sublist(1);

    // Pad sequences to blockSize if needed (for simplicity, we'll keep them shorter or truncate)
    if (input.length > blockSize) {
      input = input.sublist(0, blockSize);
      target =
          target.sublist(0, blockSize); // Make sure target matches input length
    }
    // Pad if shorter than blockSize for consistent input shapes in a batch
    while (input.length < blockSize) {
      input.add(padTokenId);
      target.add(padTokenId);
    }

    trainInputs.add(input);
    trainTargets.add(target);
  }

  print("\nDummy Training Data:");
  for (int i = 0; i < trainInputs.length; i++) {
    print("  Input:  ${trainInputs[i].map((id) => itos[id]).join(' ')}");
    print("  Target: ${trainTargets[i].map((id) => itos[id]).join(' ')}");
  }

  // 4. Instantiate the GPT model (your TransformerDecoder)
  print("\nInitializing GPT (TransformerDecoder) for training...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize:
        embedSize, // Still needed to satisfy constructor for cross-attention
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 5. Setup Optimizer
  const double learningRate = 0.01;
  final optimizer = SGD(gptModel.parameters(), learningRate);
  print("Optimizer (SGD) initialized with learning rate: $learningRate");

  // FIX: Provide a non-empty dummy encoder output to satisfy the CrossAttention layer.
  // In a true GPT, the CrossAttention layer would typically not exist or be ignored.
  // This dummy output allows the code to run without a "No element" error,
  // even though its values are not functionally meaningful for a pure GPT.
  final List<ValueVector> dummyEncoderOutput = List.generate(
    1, // Provide at least one dummy token
    (_) => ValueVector(List.filled(
        embedSize,
        Value(
            0.0))), // Each token vector should be of encoderEmbedSize (which is embedSize here)
  );

  // 6. Training Loop
  const int numEpochs = 500;
  print("\n--- Starting Training ---");

  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < trainInputs.length; i++) {
      final inputSequence = trainInputs[i];
      final targetSequence = trainTargets[i];

      // Zero gradients
      optimizer.zeroGrad();

      // Forward pass
      final List<ValueVector> logits =
          gptModel.forward(inputSequence, dummyEncoderOutput);

      // Calculate loss (Cross-Entropy Loss)
      // We are predicting the next token for each position in the input sequence.
      Value batchLoss = Value(0.0);
      int activeTokens = 0; // Count tokens that are not padding

      for (int t = 0; t < logits.length; t++) {
        // Only calculate loss for non-padding tokens
        if (targetSequence[t] != padTokenId) {
          final ValueVector tokenLogits = logits[t];
          final int trueTargetId = targetSequence[t];

          // Softmax then negative log likelihood for true target
          // This is a simplified cross-entropy calculation
          final Value trueLogit = tokenLogits.values[trueTargetId];
          final Value sumExpLogits =
              tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
          final Value logSumExp = sumExpLogits.log();
          final Value negLogProb =
              logSumExp - trueLogit; // Negative log-likelihood

          batchLoss += negLogProb;
          activeTokens++;
        }
      }

      // Average loss over active tokens
      if (activeTokens > 0) {
        batchLoss = batchLoss / Value(activeTokens.toDouble());
      } else {
        batchLoss = Value(0.0); // No active tokens, no loss
      }

      totalLoss += batchLoss.data;

      // Backward pass
      batchLoss.backward();

      // Update parameters
      optimizer.step();
    }

    if ((epoch + 1) % 1 == 0 || epoch == 0) {
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }

  print("\n--- Training Complete ---");

  // 7. Test Generation after (pseudo) training
  print("\n--- Testing Generation After Training ---");
  List<int> generatedSequence = [startTokenId];
  final int maxTestGenerationLength = 10;

  for (int i = 0; i < maxTestGenerationLength; i++) {
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    // Pass dummy encoder output as before
    final List<ValueVector> logits =
        gptModel.forward(currentInput, dummyEncoderOutput);

    // Get the logits for the last token and sample
    final ValueVector lastTokenLogits = logits.last;
    final ValueVector probabilities = lastTokenLogits.softmax();

    // Greedy sampling for simplicity
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    generatedSequence.add(predictedNextToken);

    if (predictedNextToken == stoi["."]) {
      // Stop on sentence end token
      break;
    }
    if (generatedSequence.length >= maxTestGenerationLength + 1) {
      // +1 for start token
      break;
    }
  }

  print("Generated Text: ${generatedSequence.map((id) => itos[id]).join(' ')}");
  print("---------------------------------------");
}

// file: example_gpt_training2.dart


// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
// Import your TransformerDecoder
import 'transformer_decoder.dart'; // Your existing TransformerDecoder
// Ensure all sub-modules of TransformerDecoder are accessible:
// transformer_decoder_block.dart, multi_head_attention.dart,
// feed_forward.dart, layer_norm2.dart, self_attention.dart,
// cross_attention.dart, multi_head_cross_attention.dart

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      // Only update if gradient exists
      p.data -= learningRate * p.grad;
        }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

void main() {
  print("--- Generative Pretrained Transformer (GPT) Training Example ---");

  // 1. Define GPT Model Hyperparameters
  const int vocabSize = 40; // Increased vocabulary size
  const int embedSize = 32;
  const int blockSize = 15; // Increased block size for longer sequences
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Expanded Vocabulary for demonstration
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "generation": 6,
    "model": 7,
    "the": 8,
    "quick": 9,
    "brown": 10,
    "fox": 11,
    "jumps": 12,
    "over": 13,
    "lazy": 14,
    "dog": 15,
    ".": 16,
    "<start>": 17,
    "<pad>": 18,
    "dart": 19,
    "programming": 20,
    "language": 21,
    "example": 22,
    "code": 23,
    "learning": 24,
    "machine": 25,
    "deep": 26,
    "neural": 27,
    "networks": 28,
    "great": 29,
    "simple": 30,
    "powerful": 31,
    "today": 32,
    "future": 33,
    "data": 34,
    "science": 35,
    "artificial": 36,
    "intelligence": 37,
    "next": 38,
    "token": 39,
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Verify vocabSize covers all tokens
  assert(stoi.length <= vocabSize,
      "vocabSize is too small for the defined vocabulary.");

  // Get special token IDs
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;
  final int endTokenId = stoi["."]!; // Using '.' as an example end token

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Create a Dummy Dataset with more varied sequences
  final List<List<int>> rawSequences = [
    [startTokenId, stoi["hello"]!, stoi["world"]!, stoi["."]!],
    [
      startTokenId,
      stoi["this"]!,
      stoi["is"]!,
      stoi["a"]!,
      stoi["test"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      stoi["jumps"]!,
      stoi["over"]!,
      stoi["the"]!,
      stoi["lazy"]!,
      stoi["dog"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["dart"]!,
      stoi["is"]!,
      stoi["a"]!,
      stoi["great"]!,
      stoi["programming"]!,
      stoi["language"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["learning"]!,
      stoi["deep"]!,
      stoi["neural"]!,
      stoi["networks"]!,
      stoi["is"]!,
      stoi["powerful"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["machine"]!,
      stoi["learning"]!,
      stoi["example"]!,
      stoi["code"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["artificial"]!,
      stoi["intelligence"]!,
      stoi["is"]!,
      stoi["the"]!,
      stoi["future"]!,
      stoi["."]!
    ],
  ];

  List<List<int>> trainInputs = [];
  List<List<int>> trainTargets = [];

  for (var seq in rawSequences) {
    // Input sequence: all tokens except the last one
    List<int> input = seq.sublist(0, seq.length - 1);
    // Target sequence: all tokens except the first one (what we predict)
    List<int> target = seq.sublist(1);

    // Pad or truncate sequences to blockSize
    if (input.length > blockSize) {
      input = input.sublist(0, blockSize);
      target = target.sublist(0, blockSize);
    }
    while (input.length < blockSize) {
      input.add(padTokenId);
      target.add(padTokenId);
    }

    trainInputs.add(input);
    trainTargets.add(target);
  }

  print("\nDummy Training Data:");
  for (int i = 0; i < trainInputs.length; i++) {
    print("  Input:  ${trainInputs[i].map((id) => itos[id]).join(' ')}");
    print("  Target: ${trainTargets[i].map((id) => itos[id]).join(' ')}");
  }

  // 4. Instantiate the GPT model (your TransformerDecoder)
  print("\nInitializing GPT (TransformerDecoder) for training...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 5. Setup Optimizer
  const double learningRate = 0.01;
  final optimizer = SGD(gptModel.parameters(), learningRate);
  print("Optimizer (SGD) initialized with learning rate: $learningRate");

  final List<ValueVector> dummyEncoderOutput = List.generate(
    1,
    (_) => ValueVector(List.filled(embedSize, Value(0.0))),
  );

  // 6. Training Loop
  const int numEpochs = 1000; // Increased epochs for more complex data
  print("\n--- Starting Training ---");

  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < trainInputs.length; i++) {
      final inputSequence = trainInputs[i];
      final targetSequence = trainTargets[i];

      optimizer.zeroGrad();
      final List<ValueVector> logits =
          gptModel.forward(inputSequence, dummyEncoderOutput);

      Value batchLoss = Value(0.0);
      int activeTokens = 0;

      for (int t = 0; t < logits.length; t++) {
        if (targetSequence[t] != padTokenId) {
          final ValueVector tokenLogits = logits[t];
          final int trueTargetId = targetSequence[t];

          final Value trueLogit = tokenLogits.values[trueTargetId];
          final Value sumExpLogits =
              tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
          final Value logSumExp = sumExpLogits.log();
          final Value negLogProb = logSumExp - trueLogit;

          batchLoss += negLogProb;
          activeTokens++;
        }
      }

      if (activeTokens > 0) {
        batchLoss = batchLoss / Value(activeTokens.toDouble());
      } else {
        batchLoss = Value(0.0);
      }

      totalLoss += batchLoss.data;
      batchLoss.backward();
      optimizer.step();
    }

    if ((epoch + 1) % 100 == 0 || epoch == 0) {
      // Print less frequently for more epochs
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }

  print("\n--- Training Complete ---");

  // 7. Test Generation after (pseudo) training
  print("\n--- Testing Generation After Training ---");
  List<int> generatedSequence = [startTokenId];
  final int maxTestGenerationLength = 20; // Allow longer generation

  for (int i = 0; i < maxTestGenerationLength; i++) {
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    final List<ValueVector> logits =
        gptModel.forward(currentInput, dummyEncoderOutput);

    final ValueVector lastTokenLogits = logits.last;
    final ValueVector probabilities = lastTokenLogits.softmax();

    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    generatedSequence.add(predictedNextToken);

    if (predictedNextToken == endTokenId) {
      print("End of sequence token detected.");
      break;
    }
    if (generatedSequence.length >= maxTestGenerationLength + 1) {
      print("Maximum generation length reached.");
      break;
    }
  }

  print("Generated Text: ${generatedSequence.map((id) => itos[id]).join(' ')}");
  print("---------------------------------------");
}

GPT Generation
(example_gpt_generation.dart)

This example focuses specifically on the text generation aspect of a GPT model (implemented using TransformerDecoder). It demonstrates how to use a trained (or randomly initialized) GPT to generate new sequences token by token using greedy sampling.

Key Concepts Demonstrated:

Setting up a vocabulary for token-to-ID mapping.

Initializing TransformerDecoder for generation.

Iterative token generation loop.

Greedy sampling (selecting the token with the highest probability).

Handling context window (blockSize).

// file: example_gpt_generation.dart

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';

// Import your TransformerDecoder
import 'transformer_decoder.dart'; // Your existing TransformerDecoder
// Also ensure transformer_decoder_block.dart, multi_head_attention.dart,
// feed_forward.dart, layer_norm2.dart, self_attention.dart are accessible.
// NOTE: Your TransformerDecoderBlock currently has a MultiHeadCrossAttention.
// For a pure GPT, this cross-attention layer would typically be removed
// as there's no encoder output to attend to.
// For this example, we will pass a dummy encoderOutput to satisfy the current
// TransformerDecoderBlock's forward method.
// A more accurate GPT would involve modifying transformer_decoder_block.dart
// to either remove MultiHeadCrossAttention or make it conditional.

void main() {
  print("--- Generative Pretrained Transformer (GPT) Example ---");

  // 1. Define GPT Model Hyperparameters
  const int vocabSize =
      20; // Example vocabulary size (e.g., a few common words)
  const int embedSize = 32;
  const int blockSize = 10; // Maximum sequence length the GPT can process
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Simple Vocabulary for demonstration
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "generation": 6,
    "model": 7,
    "the": 8,
    "quick": 9,
    "brown": 10,
    "fox": 11,
    "jumps": 12,
    "over": 13,
    "lazy": 14,
    "dog": 15,
    ".": 16, // End of sentence token
    "<start>": 17, // Start of sequence token
    "<pad>": 18, // Padding token
    // You might also have an <unk> token for unknown words
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Get the ID for the start token
  final int startTokenId = stoi["<start>"]!;
  final int endTokenId = stoi["."]!; // Using '.' as an example end token

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Instantiate the GPT model (your TransformerDecoder)
  print("\nInitializing GPT (TransformerDecoder)...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    // For a GPT, the cross-attention part of TransformerDecoderBlock is not used.
    // We pass embedSize here just to satisfy the constructor.
    // In a pure GPT, you'd likely have a separate TransformerDecoder class
    // that doesn't include cross-attention at all.
    encoderEmbedSize: embedSize,
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 4. Text Generation Loop (Greedy Sampling)
  print("\n--- Starting Text Generation ---");
  List<int> generatedSequence = [startTokenId]; // Start with the <start> token
  final int maxGenerationLength = 15; // Max tokens to generate

  // Create a dummy encoder output for the cross-attention layer in TransformerDecoderBlock.
  // In a true GPT, the cross-attention layer would not exist, or its input would be ignored.
  // Here, we provide an empty list or a list of zeros to prevent errors,
  // knowing that the masked self-attention is what's truly driving generation.
  final List<ValueVector> simpleDummyEncoderOutput = [
    ValueVector(List.filled(embedSize, Value(0.0)))
  ]; // (1, embedSize)

  for (int i = 0; i < maxGenerationLength; i++) {
    // If the sequence exceeds blockSize, truncate it (common for long contexts)
    // Or, for generation, keep expanding and handle attention efficiently.
    // For simplicity, we'll keep the whole generated sequence for now if within blockSize.
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    // Forward pass through the GPT (TransformerDecoder)
    // Pass the dummy encoder output to satisfy the method signature.
    final List<ValueVector> logits =
        gptModel.forward(currentInput, simpleDummyEncoderOutput);

    // Get the logits for the *last* token in the sequence (the prediction for the next token)
    final ValueVector lastTokenLogits = logits.last;

    // Apply softmax to get probabilities
    final ValueVector probabilities = lastTokenLogits.softmax();

    // Greedy sampling: pick the token with the highest probability
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    // Add the predicted token to the generated sequence
    generatedSequence.add(predictedNextToken);

    // Print current generation progress (convert IDs back to words)
    print("Generated: ${generatedSequence.map((id) => itos[id]).join(' ')}");

    // Stop if an end-of-sequence token is predicted
    if (predictedNextToken == endTokenId) {
      print("End of sequence token detected.");
      break;
    }
    if (generatedSequence.length >= maxGenerationLength + 1) {
      // +1 because we start with <start>
      print("Maximum generation length reached.");
      break;
    }
  }

  print("\n--- Final Generated Sequence ---");
  print(generatedSequence.map((id) => itos[id]).join(' '));
  print("--------------------------------");
}

Advanced Transformer Component Examples
(example3.dart)

This file contains several smaller examples demonstrating the functionality and usage of individual Transformer components like SelfAttention, MultiHeadAttention, LayerNorm, and a basic ValueMatrix (if implemented). It also includes a larger Transformer training example and a simplified sequence generation example.

Key Concepts Demonstrated:

exampleSelfAttention(): Inspecting the output shape and parameter count of a single SelfAttention head.

exampleMultiHeadAttention(): Verifying the output shape and parameter count of MultiHeadAttention.

exampleLayerNorm(): Demonstrating the normalization effect of LayerNorm.

exampleValueMatrix(): (If ValueMatrix is available) Showcasing basic matrix operations like multiplication, transpose, and scalar arithmetic.

exampleLargerTransformerTraining(): Training a Transformer model with a larger vocabulary and sequence length, similar to the basic training examples but with increased scale.

exampleSequenceGeneration(): A simplified illustration of how to generate a sequence using a decoder-only Transformer, showing the iterative process of predicting the next token.

import 'dart:math';
import 'transformer.dart';
import 'self_attention.dart';
import 'multi_head_attention.dart';
import 'layer_norm2.dart'; // Using the more concise LayerNorm
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'value_matrix.dart'; // For matrix operations if needed

/// A simple Stochastic Gradient Descent (SGD) optimizer.
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  /// Updates each parameter using its calculated gradient.
  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- Advanced Transformer Examples ---");

  // Example 1: Inspecting a single SelfAttention head
  exampleSelfAttention();

  // Example 2: Verifying MultiHeadAttention output shape
  exampleMultiHeadAttention();

  // Example 3: Demonstrating Layer Normalization
  exampleLayerNorm();

  // Example 4: Using ValueMatrix for a custom operation (if needed)
  exampleValueMatrix();

  // Example 5: Training a Transformer with a larger vocabulary and sequence
  exampleLargerTransformerTraining();

  // Example 6: Generating a sequence (simplified, as full text generation is complex)
  exampleSequenceGeneration();
}

void exampleSelfAttention() {
  print("\n--- Example 1: SelfAttention Inspection ---");

  final embedSize = 8;
  final headSize = 4;
  final sequenceLength = 3;

  final sa = SelfAttention(embedSize, headSize, masked: false);

  // Create dummy input sequence (e.g., 3 tokens, each with embedSize features)
  final x = List.generate(
      sequenceLength,
      (i) => ValueVector.fromDoubleList(
          List.generate(embedSize, (j) => Random().nextDouble())));

  print(
      "Input to SelfAttention (first token): ${x[0].values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  final output = sa.forward(x);

  print(
      "Output from SelfAttention (first token): ${output[0].values.map((v) => v.data.toStringAsFixed(2)).toList()}");
  print("SelfAttention parameters count: ${sa.parameters().length}");
}

void exampleMultiHeadAttention() {
  print("\n--- Example 2: MultiHeadAttention Shape Verification ---");

  final embedSize = 16;
  final numHeads = 4;
  final sequenceLength = 5;

  final mha = MultiHeadAttention(numHeads, embedSize, masked: true);

  final x = List.generate(
      sequenceLength,
      (i) => ValueVector.fromDoubleList(
          List.generate(embedSize, (j) => Random().nextDouble())));

  print("Input sequence length: ${x.length}");
  print("Input embedding size: ${x[0].values.length}");

  final output = mha.forward(x);

  print("Output sequence length: ${output.length}");
  print("Output embedding size: ${output[0].values.length}");
  assert(output.length == sequenceLength);
  assert(output[0].values.length == embedSize);
  print(
      "MultiHeadAttention output shape is correct: ($sequenceLength, $embedSize)");
  print("MultiHeadAttention parameters count: ${mha.parameters().length}");
}

void exampleLayerNorm() {
  print("\n--- Example 3: Layer Normalization Demonstration ---");

  final dim = 5;
  final ln = LayerNorm(dim);

  // Example vector that is not normalized
  final inputVector = ValueVector([
    Value(10.0),
    Value(20.0),
    Value(30.0),
    Value(40.0),
    Value(50.0),
  ]);

  print(
      "Input vector: ${inputVector.values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  final normalizedVector = ln.forward(inputVector);

  print(
      "Normalized vector: ${normalizedVector.values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  // To check if it's "normalized" (mean ~0, variance ~1) before gamma/beta:
  // You would need to temporarily set gamma=1, beta=0, and then calculate mean/variance of normalizedVector's data.
  // For simplicity, we just observe the output values.
  print("LayerNorm parameters count: ${ln.parameters().length}");
}

void exampleValueMatrix() {
  print("\n--- Example 4: ValueMatrix Usage ---");

  // Create two ValueMatrices
  final matrixA = ValueMatrix([
    [Value(1.0), Value(2.0)],
    [Value(3.0), Value(4.0)]
  ]);

  final matrixB = ValueMatrix([
    [Value(5.0), Value(6.0)],
    [Value(7.0), Value(8.0)]
  ]);

  print("Matrix A:\n$matrixA");
  print("Matrix B:\n$matrixB");

  // Matrix multiplication
  final product = matrixA.multiply(matrixB);
  print("A * B:\n$product");

  // Transpose
  final transposedA = matrixA.transpose();
  print("Transpose of A:\n$transposedA");

  // Scalar addition
  final scalarAdd = matrixA + Value(10.0);
  print("A + 10:\n$scalarAdd");

  // Matrix addition
  final matrixAdd = matrixA + matrixB;
  print("A + B:\n$matrixAdd");

  // Scalar multiplication
  final scalarMul = matrixA * Value(2.0);
  print("A * 2:\n$scalarMul");

  // Applying activation
  final reluA = matrixA.relu();
  print("ReLU(A):\n$reluA");
}

void exampleLargerTransformerTraining() {
  print(
      "\n--- Example 5: Training a Transformer with a larger vocabulary and sequence ---");

  final vocabSize = 50; // Increased vocabulary
  final embedSize = 32;
  final blockSize = 8; // Longer context
  final numLayers = 3;
  final numHeads = 4;

  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  final optimizer =
      SGD(model.parameters(), 0.05); // Slightly reduced learning rate

  // More complex sample data
  final sampleInputs = [0, 1, 5, 2, 8, 12, 3, 10]; // 8 tokens
  final sampleTargets = [1, 5, 2, 8, 12, 3, 10, 15]; // Next tokens for each

  final epochs = 100;
  print("\nTraining for $epochs epochs with larger data...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    final logits = model.forward(sampleInputs);

    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t];
      final targetAtT = sampleTargets[t];

      final targetVector = ValueVector(List.generate(
        vocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    final meanLoss = totalLoss / Value(logits.length.toDouble());

    model.zeroGrad();
    meanLoss.backward();
    optimizer.step();

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Larger model training complete.");
}

void exampleSequenceGeneration() {
  print("\n--- Example 6: Simplified Sequence Generation ---");

  // This is a very basic generative example. True generation involves
  // sampling from predicted probabilities and feeding the sampled token back.
  // The current model is decoder-only, so it can do this.

  final vocabSize = 10;
  final embedSize = 16;
  final blockSize = 4;

  // We'll load a pre-trained (or simply initialized) model
  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 2,
  );
  // In a real scenario, you'd load trained weights here.
  // For this example, we'll just use the randomly initialized model.

  List<int> prompt = [1, 2]; // Start with tokens 1, 2
  final int maxNewTokens = 5;

  print("Prompt: $prompt");
  print("Generating $maxNewTokens new tokens...");

  List<int> generatedSequence = List.from(prompt);

  for (int i = 0; i < maxNewTokens; i++) {
    // Crop the sequence to the block size if it exceeds it
    final currentInput = generatedSequence.length > blockSize
        ? generatedSequence.sublist(generatedSequence.length - blockSize)
        : generatedSequence;

    // Forward pass to get logits
    final logits = model.forward(currentInput);

    // Get the logits for the last token in the sequence (which is the prediction for the *next* token)
    final lastTokenLogits = logits.last;

    // Apply softmax to get probabilities
    final probabilities = lastTokenLogits.softmax();

    // Find the token with the highest probability (greedy sampling)
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    // Add the predicted token to the sequence
    generatedSequence.add(predictedNextToken);
    print(
        "Step ${i + 1}: Predicted token: $predictedNextToken (Prob: ${(maxProb * 100).toStringAsFixed(2)}%)");
  }

  print("Generated sequence: $generatedSequence");
  print(
      "Note: This is a simplified example. For better generation, consider techniques like top-k or nucleus sampling.");
}
 dart --old_gen_heap_size=6000 lib\linformer\chess_gpt_play.dart