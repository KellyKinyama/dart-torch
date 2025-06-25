import 'dart:math';
import 'dart:typed_data';

class Hp {
  int nbSteps = 1000;
  int episodeLength = 1000;
  double learningRate = 0.02;
  int nbDirections = 16;
  int nbBestDirections = 16;
  double noise = 0.03;
  int seed = 1;
  int inputSize;
  int outputSize;

  Hp(this.inputSize, this.outputSize);
}

class Normalizer {
  late int n;
  late List<double> mean;
  late List<double> meanDiff;
  late List<double> varList;

  Normalizer(int nbInputs) {
    n = 0;
    mean = List.filled(nbInputs, 0.0);
    meanDiff = List.filled(nbInputs, 0.0);
    varList = List.filled(nbInputs, 1.0);
  }

  void observe(List<double> x) {
    n += 1;
    for (int i = 0; i < x.length; i++) {
      double lastMean = mean[i];
      mean[i] += (x[i] - mean[i]) / n;
      meanDiff[i] += (x[i] - lastMean) * (x[i] - mean[i]);
      varList[i] = (meanDiff[i] / n).clamp(1e-2, double.infinity);
    }
  }

  List<double> normalize(List<double> inputs) {
    List<double> result = [];
    for (int i = 0; i < inputs.length; i++) {
      result.add((inputs[i] - mean[i]) / sqrt(varList[i]));
    }
    return result;
  }
}

class Policy {
  late List<List<double>> theta;
  final Hp hp;
  final Random rand;

  Policy(this.hp) : rand = Random(hp.seed) {
    theta = List.generate(hp.outputSize, (_) => List.filled(hp.inputSize, 0.0));
  }

  List<double> evaluate(List<double> input,
      {List<List<double>>? delta, String? direction}) {
    List<List<double>> weights;
    if (delta == null) {
      weights = theta;
    } else {
      weights = List.generate(theta.length, (i) {
        return List.generate(theta[i].length, (j) {
          double adjustment = hp.noise * delta[i][j];
          return direction == 'positive'
              ? theta[i][j] + adjustment
              : theta[i][j] - adjustment;
        });
      });
    }

    return List.generate(hp.outputSize, (i) {
      double sum = 0;
      for (int j = 0; j < hp.inputSize; j++) {
        sum += weights[i][j] * input[j];
      }
      return sum;
    });
  }

  List<List<List<double>>> sampleDeltas() {
    return List.generate(hp.nbDirections, (_) {
      return List.generate(hp.outputSize,
          (_) => List.generate(hp.inputSize, (_) => rand.nextDouble() * 2 - 1));
    });
  }

  void update(List<Tuple> rollouts, double sigmaR) {
    for (int i = 0; i < hp.outputSize; i++) {
      for (int j = 0; j < hp.inputSize; j++) {
        double step = 0.0;
        for (var rollout in rollouts) {
          step += (rollout.rPos - rollout.rNeg) * rollout.delta[i][j];
        }
        theta[i][j] +=
            (hp.learningRate / (hp.nbBestDirections * sigmaR)) * step;
      }
    }
  }
}

class Tuple {
  double rPos;
  double rNeg;
  List<List<double>> delta;

  Tuple(this.rPos, this.rNeg, this.delta);
}

class DummyEnv {
  final int inputSize;
  final int outputSize;
  final Random rand = Random(0);

  DummyEnv(this.inputSize, this.outputSize);

  List<double> reset() {
    return List.generate(inputSize, (_) => rand.nextDouble() * 2 - 1);
  }

  StepResult step(List<double> action) {
    double reward =
        -action.map((x) => x * x).reduce((a, b) => a + b); // L2 penalty
    List<double> state =
        List.generate(inputSize, (_) => rand.nextDouble() * 2 - 1);
    return StepResult(state, reward.clamp(-1, 1), false);
  }
}

class StepResult {
  final List<double> state;
  final double reward;
  final bool done;

  StepResult(this.state, this.reward, this.done);
}

double explore(DummyEnv env, Normalizer normalizer, Policy policy,
    {String? direction, List<List<double>>? delta, required Hp hp}) {
  var state = env.reset();
  double sumRewards = 0;
  for (int i = 0; i < hp.episodeLength; i++) {
    normalizer.observe(state);
    var normState = normalizer.normalize(state);
    var action = policy.evaluate(normState, delta: delta, direction: direction);
    var result = env.step(action);
    sumRewards += result.reward;
    state = result.state;
  }
  return sumRewards;
}

void train(DummyEnv env, Policy policy, Normalizer normalizer, Hp hp) {
  for (int step = 0; step < hp.nbSteps; step++) {
    var deltas = policy.sampleDeltas();
    List<double> positiveRewards = List.filled(hp.nbDirections, 0.0);
    List<double> negativeRewards = List.filled(hp.nbDirections, 0.0);

    for (int k = 0; k < hp.nbDirections; k++) {
      positiveRewards[k] = explore(env, normalizer, policy,
          direction: 'positive', delta: deltas[k], hp: hp);
    }

    for (int k = 0; k < hp.nbDirections; k++) {
      negativeRewards[k] = explore(env, normalizer, policy,
          direction: 'negative', delta: deltas[k], hp: hp);
    }

    List<double> allRewards = [...positiveRewards, ...negativeRewards];
    double mean = allRewards.reduce((a, b) => a + b) / allRewards.length;
    double sigmaR = sqrt(
        allRewards.map((r) => (r - mean) * (r - mean)).reduce((a, b) => a + b) /
            allRewards.length);

    var scores = List.generate(hp.nbDirections, (k) {
      return Tuple(positiveRewards[k], negativeRewards[k], deltas[k]);
    });

    scores.sort((a, b) => (b.rPos > b.rNeg ? b.rPos : b.rNeg)
        .compareTo(a.rPos > a.rNeg ? a.rPos : a.rNeg));

    var rollouts = scores.take(hp.nbBestDirections).toList();

    policy.update(rollouts, sigmaR);

    double rewardEval = explore(env, normalizer, policy, hp: hp);
    print('Step: $step Reward: ${rewardEval.toStringAsFixed(3)}');
  }
}

void main() {
  int inputSize = 5;
  int outputSize = 2;
  var hp = Hp(inputSize, outputSize);
  var env = DummyEnv(inputSize, outputSize);
  var policy = Policy(hp);
  var normalizer = Normalizer(inputSize);
  train(env, policy, normalizer, hp);
}
