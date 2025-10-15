import * as tf from '@tensorflow/tfjs-core';

function glorotStd(fanIn: number, fanOut: number) {
  return Math.sqrt(2 / (fanIn + fanOut));
}

function leaky<T extends tf.Tensor>(x: T, a = 0.01): T {
  return tf.tidy(() => {
    const alpha = tf.scalar(a); 
    const ax = x.mul(alpha) as T;
    return tf.maximum(x, ax) as T;
  });
}

export class IMLELabModel {
  gVariables: tf.Variable[] = [];
  gOptimizer: tf.Optimizer;
  lossType: 'L1' | 'L2';

  constructor(
    private noiseSize: number,
    private numGeneratorLayers: number,
    private numGeneratorNeurons: number,
    private noiseCoefficient: number,
    public distanceType: string,
    public epsilon: number,
  ) {
    this.distanceType = distanceType;
    this.epsilon = epsilon;
  }

  initializeModelVariables() {
    if (this.gVariables) this.gVariables.forEach(v => v.dispose());
    this.gVariables = [];

    // Generator
    const gfc0W = tf.variable(
      tf.randomNormal([this.noiseSize, this.numGeneratorNeurons], 0,
                      glorotStd(this.noiseSize, this.numGeneratorNeurons)));
    const gfc0B = tf.variable(tf.zeros([this.numGeneratorNeurons]));
    this.gVariables.push(gfc0W, gfc0B);

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = tf.variable(
      tf.randomNormal([this.numGeneratorNeurons, this.numGeneratorNeurons], 0,
                      glorotStd(this.numGeneratorNeurons, this.numGeneratorNeurons)));
      const gfcB = tf.variable(tf.zeros([this.numGeneratorNeurons]));
      this.gVariables.push(gfcW, gfcB);
    }

    const gfcLastW = tf.variable(
    tf.randomNormal([this.numGeneratorNeurons, 2], 0,
                    glorotStd(this.numGeneratorNeurons, 2)));
    const gfcLastB = tf.variable(tf.zeros([2]));
    this.gVariables.push(gfcLastW, gfcLastB);
  }

  async loadPretrainedWeights(loadedModel: tf.io.ModelArtifacts) {
    const decoded = tf.io.decodeWeights(loadedModel.weightData, loadedModel.weightSpecs);
    this.gVariables.forEach((v, i) => v.assign(decoded[`g-${i}`]));
  }

  generator(noiseTensor: tf.Tensor2D): tf.Tensor2D {
    const gfc0W = this.gVariables[0] as tf.Tensor2D;
    const gfc0B = this.gVariables[1];

    let h = leaky(noiseTensor.matMul(gfc0W).add(gfc0B));
    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const W = this.gVariables[2 + i * 2] as tf.Tensor2D;
      const B = this.gVariables[3 + i * 2];
      h = leaky(h.matMul(W).add(B));
    }

    const gfcLastW = this.gVariables[2 + this.numGeneratorLayers * 2] as tf.Tensor2D;
    const gfcLastB = this.gVariables[3 + this.numGeneratorLayers * 2];
    return h.matMul(gfcLastW).add(gfcLastB).tanh().add(tf.scalar(0.5)) as tf.Tensor2D;
  }

  nearest_neighbour(
    realData: tf.Tensor2D,         // [N_real, 2]
    generatedData: tf.Tensor2D,    // [N_gen,  2]
  ): tf.Tensor1D {
    return tf.tidy(() => {
      const realExpanded = realData.expandDims(1);        // [N_real, 1, 2]
      const genExpanded  = generatedData.expandDims(0);   // [1, N_gen, 2]
      const diff = realExpanded.sub(genExpanded);         // [N_real, N_gen, 2]

      let distances: tf.Tensor2D;
      switch (this.distanceType) {
        case 'L1':
          distances = diff.abs().sum(2) as tf.Tensor2D;
          break;
        case 'Barrier': {
          const lam = tf.scalar(1e-3);
          const eps = tf.scalar(1e-8);
          const r = diff.square().sum(2).sqrt();
          distances = r.add(lam.div(r.add(eps))) as tf.Tensor2D;
          break;
        }
        default:
          distances = diff.square().sum(2) as tf.Tensor2D; // L2 (squared)
      }

      // RS-IMLE: reject generators too close to ANY real point
      const minDistToReal = distances.min(0) as tf.Tensor1D;                    // [N_gen]
      let keepMask = minDistToReal.greater(tf.scalar(this.epsilon)) as tf.Tensor1D; // [N_gen] bool

      // Ensure at least one survivor: force-keep farthest generator (no squeeze/oneHot)
      const farIdxScalar = minDistToReal.argMax();        // R0
      const depth = generatedData.shape[0] | 0;           // N_gen (int)
      const forceKeep = tf.range(0, depth, 1, 'int32')
                        .equal(farIdxScalar.toInt()) as tf.Tensor1D; // [N_gen] bool
      keepMask = keepMask.logicalOr(forceKeep) as tf.Tensor1D;        // [N_gen] bool

      // Mask rejected columns with a huge value; broadcast mask to [N_real, N_gen]
      const large = tf.scalar(1e9);
      const mask2D = keepMask.logicalNot().toFloat().expandDims(0);   // [1, N_gen]
      const maskedDistances = distances.add(mask2D.mul(large)) as tf.Tensor2D;

      return maskedDistances.argMin(1) as tf.Tensor1D;                // [N_real]
    });
  }

  imleLoss(realData: tf.Tensor2D, matchedNoise: tf.Tensor2D): tf.Scalar {
    const perturbation = tf.randomNormal(matchedNoise.shape).mul(tf.scalar(this.noiseCoefficient));
    const perturbedLatents = matchedNoise.add(perturbation) as tf.Tensor2D;
    const generated = this.generator(perturbedLatents);

    return realData.sub(generated).square().mean() as tf.Scalar;
  }

  updateOptimizer(optimizerType: string, learningRate: number) {
    switch (optimizerType) {
    case 'Adam':
      this.gOptimizer = tf.train.adam(learningRate, 0.9, 0.999);
      break;
    case 'Adagrad':
      this.gOptimizer = tf.train.adagrad(learningRate);
      break;
    case 'RMSProp':
      this.gOptimizer = tf.train.rmsprop(learningRate, 0.9, 0.0, 1e-8, false);
      break;
    default:
      this.gOptimizer = tf.train.sgd(learningRate);
  }
  }
}
