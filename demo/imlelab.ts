import * as d3 from 'd3-selection';
import { scaleSequential } from 'd3-scale';
import { interpolateGreens, interpolatePRGn } from 'd3-scale-chromatic';
import { line } from 'd3-shape';
import * as d3Transition from 'd3-transition';

import { PolymerElement, PolymerHTMLElement } from '../lib/polymer-spec';
import * as tf from '@tensorflow/tfjs-core';

import * as ganlab_input_providers from './ganlab_input_providers';
import * as ganlab_drawing from './ganlab_drawing';
import * as ganlab_evaluators from './ganlab_evaluators';
import * as imlelab_models from './imle_model';

declare const Chart: any;
type ChartPoint = { x: number; y: number | null };
type ChartData = ChartPoint[];

/* ----------------------------------------------------------------------------
   Constants
---------------------------------------------------------------------------- */
const BATCH_SIZE = 150;
const ATLAS_SIZE = 12000;

const NUM_GRID_CELLS = 30;
const NUM_MANIFOLD_CELLS = 20;
const GRAD_ARROW_UNIT_LEN = 0.15;
const NUM_TRUE_SAMPLES_VISUALIZED = 450;

const VIS_INTERVAL = 50;
const EPOCH_INTERVAL = 2;
const SLOW_INTERVAL_MS = 1250;

interface ManifoldCell {
  points: Float32Array[];
  area?: number;
}

// tslint:disable-next-line:variable-name
const GANLabPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'gan-lab',
  properties: {
    dLearningRate: Number, // kept for UI compat; not used in IMLE
    gLearningRate: Number,
    learningRateOptions: Array,
    dOptimizerType: String, // kept for UI compat; not used in IMLE
    gOptimizerType: String,
    optimizerTypeOptions: Array,
    lossType: String,
    lossTypeOptions: Array,
    selectedShapeName: String,
    shapeNames: Array,
    selectedNoiseType: String,
    noiseTypes: Array,
    epsilonType: Number,
    epsilonTypeOptions: Array,
  }
});

class GANLab extends GANLabPolymer {
  private iterationCount: number;

  private noiseProvider!: ganlab_input_providers.InputProvider;
  private noiseProviderFixed!: ganlab_input_providers.InputProvider;
  private trueSampleProvider!: ganlab_input_providers.InputProvider;
  private trueSampleProviderFixed!: ganlab_input_providers.InputProvider;
  private uniformNoiseProvider!: ganlab_input_providers.InputProvider;
  private uniformInputProvider!: ganlab_input_providers.InputProvider;

  private usePretrained: boolean;

  private model!: imlelab_models.IMLELabModel;
  private noiseSize: number;
  private numGeneratorLayers: number;
  private numDiscriminatorLayers: number;   // kept for UI compat
  private numGeneratorNeurons: number;
  private numDiscriminatorNeurons: number;  // kept for UI compat
  private kDSteps: number;                  // kept for UI compat (unused)
  private kGSteps: number;
  private sampleFactor: number;
  private noiseCoefficient: number;
  private epsilon: number;

  private plotSizePx: number;
  private mediumPlotSizePx: number;
  private smallPlotSizePx: number;
  private densitiesForGaussian!: number[];

  private gDotsElementList: string[];
  private highlightedComponents!: HTMLDivElement[];
  private highlightedTooltip!: HTMLDivElement;

  private evaluator!: ganlab_evaluators.GANLabEvaluatorGridDensities;

  private canvas!: HTMLCanvasElement;
  private drawing!: ganlab_drawing.GANLabDrawing;

  private stepMode: boolean;
  private slowMode: boolean;
  private isPlaying: boolean;
  private isPausedOngoingIteration: boolean;
  private iterCountElement!: HTMLElement;

  private dFlowElements!: NodeListOf<SVGPathElement>; // UI lines
  private gFlowElements!: NodeListOf<SVGPathElement>;
  private finishDrawingButton!: HTMLInputElement;

  private costChartData!: ChartData[];
  private costChart: any;
  private evalChartData!: ChartData[];
  private evalChart: any;

  // configurable NN metric; UI toggle can be added later
  private distanceType: 'L1' | 'L2' = 'L2';

  // ───────────────────────────────────────────────────────────────────────────
  // Uniform scaling helpers
  // ───────────────────────────────────────────────────────────────────────────
  private xPx(x: number, size: number) { return x * size; }
  private yPx(y: number, size: number) { return (1 - y) * size; }
  private sizeFor(sel: string): number {
    const big = new Set([
      '#vis-generated-samples',
      '#vis-manifold',
      '#vis-generator-gradients',
      '#vis-true-samples',
      '#vis-discriminator-output',
    ]);
    const medium = new Set(['#svg-generator-manifold', '#svg-discriminator-output']);
    if (big.has(sel)) return this.plotSizePx;
    if (medium.has(sel)) return this.mediumPlotSizePx;
    return this.smallPlotSizePx;
  }

  private getLatentPoolFromFixedProvider(factor: number): tf.Tensor2D {
    const batches: tf.Tensor2D[] = [];
    for (let i = 0; i < factor; ++i) {
      batches.push(this.noiseProviderFixed.getNextCopy() as tf.Tensor2D);
    }
    return tf.concat(batches, 0) as tf.Tensor2D;
  }

  ready() {
    // HTML elements for generator architecture
    const numGeneratorLayersElement =
      document.getElementById('num-g-layers') as HTMLElement;
    this.numGeneratorLayers = +numGeneratorLayersElement.innerText;
    document.getElementById('g-layers-add-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers < 5) {
          this.numGeneratorLayers += 1;
          numGeneratorLayersElement.innerText = this.numGeneratorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });
    document.getElementById('g-layers-remove-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers > 0) {
          this.numGeneratorLayers -= 1;
          numGeneratorLayersElement.innerText = this.numGeneratorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numGeneratorNeuronsElement =
      document.getElementById('num-g-neurons') as HTMLElement;
    this.numGeneratorNeurons = +numGeneratorNeuronsElement.innerText;
    document.getElementById('g-neurons-add-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons < 100) {
          this.numGeneratorNeurons += 1;
          numGeneratorNeuronsElement.innerText = this.numGeneratorNeurons.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });
    document.getElementById('g-neurons-remove-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons > 0) {
          this.numGeneratorNeurons -= 1;
          numGeneratorNeuronsElement.innerText = this.numGeneratorNeurons.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numKGStepsElement =
      document.getElementById('k-g-steps') as HTMLElement;
    this.kGSteps = +numKGStepsElement.innerText;
    document.getElementById('k-g-steps-add-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps < 10) {
          this.kGSteps += 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });
    document.getElementById('k-g-steps-remove-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps > 0) {
          this.kGSteps -= 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });

    // Distance
    this.distanceTypeOptions = ['L1', 'L2', 'Barrier'];
    this.distanceType = 'L2';
    this.querySelector('#distance-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.distanceType = event.detail.selected;
        this.model.distanceType = this.distanceType;
      });

    // noiseScale
    this.noiseScaleTypeOptions = [0.001, 0.01, 0.1];
    this.noiseCoefficient = 0.001;
    this.querySelector('#noise-scale-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.noiseCoefficient = event.detail.selected;
      });

    // sample factor
    this.sampleFactorTypeOptions = [1, 2, 4, 8, 16, 32];
    this.sampleFactor = 1;
    this.querySelector('#sample-factor-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.sampleFactor = event.detail.selected;
      });

    this.epsilonTypeOptions = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01];
    this.epsilon = 0;
    this.querySelector('#epsilon-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.epsilon= event.detail.selected;
      });

    this.learningRateOptions = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];
    this.gLearningRate = 0.1;
    this.querySelector('#g-learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.gLearningRate = +event.detail.selected;
        this.model.updateOptimizer(this.gOptimizerType, this.gLearningRate);
      });

    this.optimizerTypeOptions = ['SGD', 'Adam', 'Adagrad', 'RMSProp'];
    this.gOptimizerType = 'SGD';
    this.querySelector('#g-optimizer-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.gOptimizerType = event.detail.selected;
        this.model.updateOptimizer(this.gOptimizerType, this.gLearningRate);
      });

    this.shapeNames = ['line', 'gaussians', 'ring', 'disjoint', 'drawing'];
    this.selectedShapeName = 'gaussians';

    const distributionElementList = document.querySelectorAll('.distribution-item');
    for (let i = 0; i < distributionElementList.length; ++i) {
      // tslint:disable-next-line:no-any
      distributionElementList[i].addEventListener('click', (event: any) =>
        this.changeDataset(event.target), false);
    }

    this.noiseTypes = ['1D Gaussian', '2D Gaussian'];
    // this.noiseTypes = ['1D Uniform', '1D Gaussian', '2D Uniform', '2D Gaussian'];
    this.selectedNoiseType = '2D Gaussian';
    this.noiseSize = 2;
    this.querySelector('#noise-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any
      'iron-activate', (event: any) => {
        this.selectedNoiseType = event.detail.selected;
        this.noiseSize = +this.selectedNoiseType.substring(0, 1);
        this.disabledPretrainedMode();
        this.createExperiment();
      });

    // Checkbox toggles (visibility of layers/panels)
    const checkboxList = [
      {
        graph: '#overlap-plots',
        description: '#toggle-right-discriminator',
        layer: '#vis-discriminator-output'
      },
      {
        graph: '#enable-manifold',
        description: '#toggle-right-generator',
        layer: '#vis-manifold'
      },
      {
        graph: '#show-t-samples',
        description: '#toggle-right-real-samples',
        layer: '#vis-true-samples'
      },
      {
        graph: '#show-g-samples',
        description: '#toggle-right-fake-samples',
        layer: '#vis-generated-samples'
      },
      {
        graph: '#show-g-gradients',
        description: '#toggle-right-gradients',
        layer: '#vis-generator-gradients'
      }
    ];
    checkboxList.forEach(layer => {
      this.querySelector(layer.graph)!.addEventListener(
        'change', (event: Event) => {
          const container = this.querySelector(layer.layer) as SVGGElement;
          // tslint:disable-next-line:no-any
          container.style.visibility = (event.target as any).checked ? 'visible' : 'hidden';
          const element = this.querySelector(layer.description) as HTMLElement;
          if ((event.target as any).checked) {
            element.classList.add('checked');
          } else {
            element.classList.remove('checked');
          }
        });
      this.querySelector(layer.description)!.addEventListener(
        'click', (event: Event) => {
          const spanElement = this.querySelector(layer.description) as HTMLElement;
          const container = this.querySelector(layer.layer) as HTMLElement;
          const element = this.querySelector(layer.graph) as HTMLInputElement;

          if ((event.target as any).classList.contains('checked')) {
            spanElement.classList.remove('checked');
            container.style.visibility = 'hidden';
            element.checked = false;
          } else {
            spanElement.classList.add('checked');
            container.style.visibility = 'visible';
            element.checked = true;
          }
        });
    });

    // Pre-trained checkbox
    this.usePretrained = true;
    this.querySelector('#toggle-pretrained')!.addEventListener(
      'change', (event: Event) => {
        // tslint:disable-next-line:no-any
        this.usePretrained = (event.target as any).checked;
        this.loadModelAndCreateExperiment();
      });

    // Timeline controls
    document.getElementById('play-pause-button')!.addEventListener(
      'click', () => this.onClickPlayPauseButton());
    document.getElementById('reset-button')!.addEventListener(
      'click', () => this.onClickResetButton());
    document.getElementById('next-step-g-button')!.addEventListener(
      'click', () => this.onClickNextStepButton('G'));

    this.stepMode = false;
    document.getElementById('next-step-button')!.addEventListener(
      'click', () => this.onClickStepModeButton());

    this.slowMode = false;
    document.getElementById('slow-step')!.addEventListener(
      'click', () => this.onClickSlowModeButton());

    this.editMode = true;
    document.getElementById('edit-model-button')!.addEventListener(
      'click', () => this.onClickEditModeButton());
    this.onClickEditModeButton();

    this.iterCountElement = document.getElementById('iteration-count') as HTMLElement;

    document.getElementById('save-model')!.addEventListener(
      'click', () => this.onClickSaveModelButton());

    // Visualization sizes
    this.plotSizePx = 400;
    this.mediumPlotSizePx = 140;
    this.smallPlotSizePx = 50;

    this.colorScale = interpolatePRGn;

    // Reserve big plot for generated samples
    this.gDotsElementList = ['#vis-generated-samples'];

    this.dFlowElements =
      this.querySelectorAll('.d-update-flow') as NodeListOf<SVGPathElement>;
    this.gFlowElements =
      this.querySelectorAll('.g-update-flow') as NodeListOf<SVGPathElement>;

    // Generator manifold hover animation
    document.getElementById('svg-generator-manifold')!.addEventListener(
      'mouseenter', () => {
        this.playGeneratorAnimation();
      });

    // Drawing-related
    this.canvas = document.getElementById('input-drawing-canvas') as HTMLCanvasElement;
    this.drawing = new ganlab_drawing.GANLabDrawing(this.canvas, this.plotSizePx);

    this.finishDrawingButton =
      document.getElementById('finish-drawing') as HTMLInputElement;
    this.finishDrawingButton.addEventListener(
      'click', () => this.onClickFinishDrawingButton());

    // Create a new experiment
    this.loadModelAndCreateExperiment();
  }

  private createExperiment() {
    // Reset.
    this.pause();
    this.iterationCount = 0;
    this.iterCountElement.innerText = this.zeroPad(this.iterationCount);

    this.isPausedOngoingIteration = false;

    document.getElementById('g-loss-value')!.innerText = '-';
    document.getElementById('g-loss-bar')!.style.width = '0';
    this.recreateCharts();

    const dataElements = [
      d3.select('#vis-true-samples').selectAll('.true-dot'),
      d3.select('#svg-true-samples').selectAll('.true-dot'),
      d3.select('#svg-true-prediction').selectAll('.true-dot'),
      d3.select('#svg-noise').selectAll('.noise-dot'),
      d3.select('#vis-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-prediction').selectAll('.generated-dot'),
      d3.select('#vis-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#svg-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#vis-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#vis-manifold').selectAll('.manifold-cells'),
      d3.select('#vis-manifold').selectAll('.grids'),
      d3.select('#svg-generator-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#svg-generator-manifold').selectAll('.manifold-cells'),
      d3.select('#svg-generator-manifold').selectAll('.grids'),
      d3.select('#vis-generator-gradients').selectAll('.gradient-generated'),
      d3.select('#svg-generator-gradients').selectAll('.gradient-generated'),
      d3.select('#svg-discriminator-output').selectAll('.matching-line'),
      d3.select('#vis-discriminator-output').selectAll('.matching-line'),
    ];
    dataElements.forEach((element) => {
      element.data([]).exit().remove();
    });

    // Input providers
    const noiseProviderBuilder =
      new ganlab_input_providers.GANLabNoiseProviderBuilder(
        this.noiseSize, this.selectedNoiseType,
        ATLAS_SIZE, BATCH_SIZE);
    noiseProviderBuilder.generateAtlas();
    this.noiseProvider = noiseProviderBuilder.getInputProvider();
    this.noiseProviderFixed = noiseProviderBuilder.getInputProvider(true);

    const drawingPositions = this.drawing.drawingPositions;
    const trueSampleProviderBuilder =
      new ganlab_input_providers.GANLabTrueSampleProviderBuilder(
        ATLAS_SIZE, this.selectedShapeName,
        drawingPositions, BATCH_SIZE);
    trueSampleProviderBuilder.generateAtlas();
    this.trueSampleProvider = trueSampleProviderBuilder.getInputProvider();
    this.trueSampleProviderFixed =
      trueSampleProviderBuilder.getInputProvider(true);

    if (this.noiseSize <= 2) {
      const uniformNoiseProviderBuilder =
        new ganlab_input_providers.GANLabUniformNoiseProviderBuilder(
          this.noiseSize, NUM_MANIFOLD_CELLS, BATCH_SIZE);
      uniformNoiseProviderBuilder.generateAtlas();
      if (this.selectedNoiseType === '2D Gaussian') {
        this.densitiesForGaussian =
          uniformNoiseProviderBuilder.calculateDensitiesForGaussian();
      }
      this.uniformNoiseProvider =
        uniformNoiseProviderBuilder.getInputProvider();
    }

    const uniformSampleProviderBuilder =
      new ganlab_input_providers.GANLabUniformSampleProviderBuilder(
        NUM_GRID_CELLS, BATCH_SIZE);
    uniformSampleProviderBuilder.generateAtlas();
    this.uniformInputProvider = uniformSampleProviderBuilder.getInputProvider();

    // Visualize true samples/noise
    this.visualizeTrueDistribution(trueSampleProviderBuilder.getInputAtlas());
    this.visualizeNoiseDistribution(noiseProviderBuilder.getNoiseSample());

    // Evaluator
    this.evaluator =
      new ganlab_evaluators.GANLabEvaluatorGridDensities(NUM_GRID_CELLS);
    this.evaluator.createGridsForTrue(
      trueSampleProviderBuilder.getInputAtlas(), NUM_TRUE_SAMPLES_VISUALIZED);

    // Model
    this.model = new imlelab_models.IMLELabModel(
      this.noiseSize, this.numGeneratorLayers, this.numGeneratorNeurons, this.noiseCoefficient, this.distanceType, this.epsilon);
    this.model.initializeModelVariables();
    this.model.updateOptimizer(this.gOptimizerType, this.gLearningRate);
  }

  private changeDataset(element: HTMLElement) {
    this.selectedShapeName = element.getAttribute('data-distribution-name')!;

    const distributionElementList = document.querySelectorAll('.distribution-item');
    for (let i = 0; i < distributionElementList.length; ++i) {
      if (distributionElementList[i].classList.contains('selected')) {
        distributionElementList[i].classList.remove('selected');
      }
    }
    if (!element.classList.contains('selected')) {
      element.classList.add('selected');
    }

    this.disabledPretrainedMode();
    this.loadModelAndCreateExperiment();
  }

  private loadModelAndCreateExperiment() {
    if (this.selectedShapeName === 'drawing') {
      this.pause();
      this.drawing.prepareDrawing();
      this.disabledPretrainedMode();
    } else if (this.usePretrained === true) {
      const filename = `pretrained_${this.selectedShapeName}`;
      this.loadPretrainedWeightFile(filename).then((loadedModel) => {
        const loadedIterCount = this.iterationCount;
        this.createExperiment();
        this.model.loadPretrainedWeights(loadedModel);

        // Run one iteration for visualization.
        this.isPlaying = true;
        this.iterateTraining(false);
        this.isPlaying = false;

        this.iterationCount = loadedIterCount;
        this.iterCountElement.innerText = this.zeroPad(this.iterationCount);
      });
    } else {
      const filename = `pretrained_${this.selectedShapeName}`;
      this.loadPretrainedWeightFile(filename).then((_loadedModel) => {
        this.createExperiment();
      });
    }
  }

  private visualizeTrueDistribution(inputAtlasList: number[]) {
    const color = scaleSequential(interpolateGreens).domain([0, 0.05]);

    const trueDistribution: Array<[number, number]> = [];
    while (trueDistribution.length < NUM_TRUE_SAMPLES_VISUALIZED) {
      const values = inputAtlasList.splice(0, 2);
      trueDistribution.push([values[0], values[1]]);
    }

    const trueDotsElementList = ['#vis-true-samples', '#svg-true-samples'];
    trueDotsElementList.forEach((sel) => {
      const size = this.sizeFor(sel);
      const radius = (size === this.plotSizePx) ? 2 : 1;
      d3.select(sel)
        .selectAll('.true-dot')
        .data(trueDistribution)
        .enter()
        .append('circle')
        .attr('class', 'true-dot gan-lab')
        .attr('r', radius)
        .attr('cx', (d: number[]) => this.xPx(d[0], size))
        .attr('cy', (d: number[]) => this.yPx(d[1], size))
        .append('title')
        .text((d: number[]) => `${d[0].toFixed(2)}, ${d[1].toFixed(2)}`);
    });
  }

  private visualizeNoiseDistribution(inputList: Float32Array) {
    const noiseSamples: number[][] = [];
    for (let i = 0; i < inputList.length / this.noiseSize; ++i) {
      const values: number[] = [];
      for (let j = 0; j < this.noiseSize; ++j) {
        values.push(inputList[i * this.noiseSize + j]);
      }
      noiseSamples.push(values);
    }

    const size = this.sizeFor('#svg-noise');
    d3.select('#svg-noise')
      .selectAll('.noise-dot')
      .data(noiseSamples)
      .enter()
      .append('circle')
      .attr('class', 'noise-dot gan-lab')
      .attr('r', 1)
      .attr('cx', (d: number[]) => this.xPx(d[0], size))
      .attr('cy', (d: number[]) => this.noiseSize === 1
        ? size / 2
        : this.yPx(d[1], size))
      .append('title')
      .text((d: number[], i: number) => this.noiseSize === 1
        ? `${Number(d[0]).toFixed(2)} (${i})`
        : `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)} (${i})`);
  }

  private onClickFinishDrawingButton() {
    if (this.drawing.drawingPositions.length === 0) {
      alert('Draw something on canvas');
    } else {
      const drawingElement = this.querySelector('#drawing-container') as HTMLElement;
      drawingElement.style.display = 'none';
      const drawingBackgroundElement =
        this.querySelector('#drawing-disable-background') as HTMLDivElement;
      drawingBackgroundElement.style.display = 'none';
      this.createExperiment();
    }
  }

  private disabledPretrainedMode() {
    this.usePretrained = false;
    const element = document.getElementById('toggle-pretrained') as HTMLInputElement;
    element.checked = false;
  }

  private play() {
    if (this.stepMode) this.onClickStepModeButton();
    this.isPlaying = true;
    document.getElementById('play-pause-button')!.classList.add('playing');
    if (!this.isPausedOngoingIteration) {
      this.iterateTraining(true);
    }
    document.getElementById('model-vis-svg')!.classList.add('playing');
  }

  private pause() {
    this.isPlaying = false;
    const button = document.getElementById('play-pause-button')!;
    if (button.classList.contains('playing')) {
      button.classList.remove('playing');
    }
    document.getElementById('model-vis-svg')!.classList.remove('playing');
  }

  private onClickPlayPauseButton() {
    if (this.isPlaying) this.pause(); else this.play();
  }

  private onClickNextStepButton(type?: string) {
    if (this.isPlaying) this.pause();
    this.isPlaying = true;
    this.iterateTraining(false, type);
    this.isPlaying = false;
  }

  private onClickResetButton() {
    if (this.isPlaying) this.pause();
    this.loadModelAndCreateExperiment();
  }

  private onClickStepModeButton() {
    if (!this.stepMode) {
      if (this.isPlaying) this.pause();
      if (this.slowMode) this.onClickSlowModeButton();

      this.stepMode = true;
      document.getElementById('next-step-button')!
        .classList.add('mdl-button--colored');
      document.getElementById('step-buttons')!.style.display = 'block';
    } else {
      this.stepMode = false;
      document.getElementById('next-step-button')!
        .classList.remove('mdl-button--colored');
      document.getElementById('step-buttons')!.style.display = 'none';
    }
  }

  private onClickSlowModeButton() {
    if (this.editMode) this.onClickEditModeButton();
    this.slowMode = !this.slowMode;

    if (this.slowMode === true) {
      if (this.stepMode) this.onClickStepModeButton();
      document.getElementById('slow-step')!
        .classList.add('mdl-button--colored');
      document.getElementById('tooltips')!.classList.add('shown');
    } else {
      document.getElementById('slow-step')!
        .classList.remove('mdl-button--colored');
      this.dehighlightStep();
      const container = document.getElementById('model-visualization-container')!;
      if (container.classList.contains('any-highlighted')) {
        container.classList.remove('any-highlighted');
      }
      document.getElementById('component-generator')!.classList.remove('deactivated');
      document.getElementById('component-discriminator')!.classList.remove('deactivated');
      document.getElementById('component-g-loss')!.classList.remove('activated');
      for (let i = 0; i < this.gFlowElements.length; ++i) {
        this.gFlowElements[i].classList.remove('g-activated');
      }
      document.getElementById('tooltips')!.classList.remove('shown');
    }
  }

  private onClickEditModeButton() {
    const elements: NodeListOf<HTMLDivElement> =
      this.querySelectorAll('.config-item');
    for (let i = 0; i < elements.length; ++i) {
      elements[i].style.visibility = this.editMode ? 'hidden' : 'visible';
    }
    this.editMode = !this.editMode;
    if (this.editMode === true) {
      document.getElementById('edit-model-button')!
        .classList.add('mdl-button--colored');
    } else {
      document.getElementById('edit-model-button')!
        .classList.remove('mdl-button--colored');
    }
  }

  private zeroPad(n: number): string {
    const pad = '000000';
    return (pad + n).slice(-pad.length).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  private async iterateTraining(keepIterating: boolean, type?: string) {
    if (!this.isPlaying) return;

    this.iterationCount++;

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % EPOCH_INTERVAL === 0) {
      this.iterCountElement.innerText = this.zeroPad(this.iterationCount);

      d3.select('#model-vis-svg')
        .selectAll('path')
        .style('stroke-dashoffset', () => this.iterationCount * (-1));
    }

    // Visualize generated samples before training.
    if (this.slowMode) {
      const container = document.getElementById('model-visualization-container')!;
      if (!container.classList.contains('any-highlighted')) {
        container.classList.add('any-highlighted');
      }

      await this.sleep(SLOW_INTERVAL_MS);

      await this.highlightIMLEStep(
        ['component-noise', 'component-generator', 'component-generated-samples'],
        'tooltip-generated-samples');
    }

    // show generated samples (fixed noise) pre-update
    tf.tidy(() => {
      if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
          this.iterationCount % VIS_INTERVAL === 0) {
        const gDataBefore: Array<[number, number]> = [];
        const noiseFixedBatch = this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gResult = this.model.generator(noiseFixedBatch);
        const gResultData = gResult.dataSync() as Float32Array;
        for (let j = 0; j < gResultData.length / 2; ++j) {
          gDataBefore.push([gResultData[j * 2], gResultData[j * 2 + 1]]);
        }

        if (this.iterationCount === 1) {
          this.gDotsElementList.forEach((sel) => {
            const size = this.sizeFor(sel);
            const radius = (size === this.plotSizePx) ? 2 : 1;
            d3.select(sel).selectAll('.generated-dot')
              .data(gDataBefore)
              .enter()
              .append('circle')
              .attr('class', 'generated-dot gan-lab')
              .attr('r', radius)
              .attr('cx', (d: number[]) => this.xPx(d[0], size))
              .attr('cy', (d: number[]) => this.yPx(d[1], size))
              .append('title')
              .text((d: number[]) =>
                `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)}`);
          });
        } else {
          this.gDotsElementList.forEach((sel) => {
            const size = this.sizeFor(sel);
            d3Transition.transition()
              .select(sel)
              .selectAll('.generated-dot')
              .selection().data(gDataBefore)
              .transition().duration(SLOW_INTERVAL_MS / 600)
              .attr('cx', (d: number[]) => this.xPx(d[0], size))
              .attr('cy', (d: number[]) => this.yPx(d[1], size));
          });
        }
      }
    });

    if (this.slowMode) {
      await this.highlightIMLEStep(
        ['component-true-samples', 'component-generated-samples', 'component-discriminator'],
        'tooltip-imle-matching');
    }

    // ---------------------------
    // NN rendering (real ↔ gen)
    // ---------------------------
    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % VIS_INTERVAL === 0) {
      tf.tidy(() => {
        // 1) Real batch used for matching (fixed provider so it's stable within frame)
        const trueSampleBatch =
          this.trueSampleProviderFixed.getNextCopy() as tf.Tensor2D; // [B,2]
        const realArr = trueSampleBatch.dataSync() as Float32Array;
        const realPoints: Array<[number, number]> = [];
        for (let i = 0; i < realArr.length; i += 2) {
          realPoints.push([realArr[i], realArr[i + 1]]);
        }

        // 2) Candidate pool -> generated points
        const latents = this.getLatentPoolFromFixedProvider(this.sampleFactor);
        const generatedData = this.model.generator(latents) as tf.Tensor2D;
        const genArr = generatedData.dataSync() as Float32Array;

        const candidatePoints: Array<[number, number]> = [];
        for (let i = 0; i < genArr.length; i += 2) {
          candidatePoints.push([genArr[i], genArr[i + 1]]);
        }

        // 3) Nearest neighbours: real -> generated  (force int32 dtype)
        const nearestIndices = this.model.nearest_neighbour(
          trueSampleBatch, generatedData
        ).toInt() as tf.Tensor1D;

        const idxArr = nearestIndices.dataSync() as Int32Array;

        // 4) Build pairs (for line/gradient drawing)
        type NNPair = { real: [number, number]; gen: [number, number] };
        const pairs: NNPair[] = [];
        const gradData: [number, number, number, number][] = [];

        for (let i = 0; i < idxArr.length; i++) {
          const ri = i * 2;
          const gi = idxArr[i] * 2;
          const real = [realArr[ri], realArr[ri + 1]] as [number, number];
          const gen  = [genArr[gi],  genArr[gi + 1]] as [number, number];
          pairs.push({ real, gen });
          // IMLE "gradient arrow": from gen → real
          gradData.push([gen[0], gen[1], real[0] - gen[0], real[1] - gen[1]]);
        }

        // -------- Render NN lines IN the NN panel --------
        const nnSel = '#svg-discriminator-output';
        const sizeNN = this.sizeFor(nnSel);
        const nnSvg = d3.select(nnSel);
        const LINE_COLOR = '#4b82cfff';

        const lSel = nnSvg
          .selectAll<SVGLineElement, NNPair>('.matching-line')
          .data(pairs);

        lSel.exit().remove();

        const lEnter = lSel.enter()
          .append('line')
          .attr('class', 'matching-line gan-lab')
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.4)
          .attr('stroke', LINE_COLOR);

        (lSel as any).merge(lEnter)
          .attr('x1', d => this.xPx(d.real[0], sizeNN))
          .attr('y1', d => this.yPx(d.real[1], sizeNN))
          .attr('x2', d => this.xPx(d.gen[0],  sizeNN))
          .attr('y2', d => this.yPx(d.gen[1],  sizeNN));

        // Also draw the same NN lines on the layered distribution (big) layer.
        {
          const bigSel = '#vis-discriminator-output';
          const bigSize = this.sizeFor(bigSel);
          const bigG = d3.select(bigSel);

          const bigLines = bigG
            .selectAll<SVGLineElement, NNPair>('.matching-line')
            .data(pairs);

          bigLines.exit().remove();

          const bigEnter = bigLines.enter()
            .append('line')
            .attr('class', 'matching-line gan-lab')
            .attr('stroke-width', 1.5)
            .attr('opacity', 0.4)
            .attr('stroke', '#4b82cfff')
            .attr('pointer-events', 'none');

          (bigLines as any).merge(bigEnter)
            .attr('x1', d => this.xPx(d.real[0], bigSize))
            .attr('y1', d => this.yPx(d.real[1], bigSize))
            .attr('x2', d => this.xPx(d.gen[0],  bigSize))
            .attr('y2', d => this.yPx(d.gen[1],  bigSize));

          bigG.selectAll('.matching-line').raise();
        }



        // -------- Small right panels (dots only) --------
        {
          const sSel = '#svg-generated-samples';
          const sSize = this.sizeFor(sSel);
          const s = d3.select(sSel)
            .selectAll<SVGCircleElement, [number, number]>('.generated-dot')
            .data(candidatePoints);
          if (this.iterationCount === 1) {
            s.enter().append('circle')
              .attr('class', 'generated-dot gan-lab')
              .attr('r', 1)
              .merge(s as any)
              .attr('cx', d => this.xPx(d[0], sSize))
              .attr('cy', d => this.yPx(d[1], sSize));
          } else {
            s.enter().append('circle').attr('class', 'generated-dot gan-lab').attr('r', 1);
            d3.select(sSel)
              .selectAll('.generated-dot')
              .data(candidatePoints)
              .attr('cx', d => this.xPx(d[0], sSize))
              .attr('cy', d => this.yPx(d[1], sSize));
          }
        }

        {
          const pSel = '#svg-generated-prediction';
          const pSize = this.sizeFor(pSel);
          const selectedPoints: Array<[number, number]> = [];
          for (let i = 0; i < idxArr.length; i++) {
            const gi = idxArr[i] * 2;
            selectedPoints.push([genArr[gi], genArr[gi + 1]]);
          }

          const s2 = d3.select(pSel)
            .selectAll<SVGCircleElement, [number, number]>('.generated-dot')
            .data(selectedPoints);
          if (this.iterationCount === 1) {
            s2.enter().append('circle')
              .attr('class', 'generated-dot gan-lab')
              .attr('r', 1)
              .merge(s2 as any)
              .attr('cx', d => this.xPx(d[0], pSize))
              .attr('cy', d => this.yPx(d[1], pSize));
          } else {
            s2.enter().append('circle').attr('class', 'generated-dot gan-lab').attr('r', 1);
            d3.select(pSel)
              .selectAll('.generated-dot')
              .data(selectedPoints)
              .attr('cx', d => this.xPx(d[0], pSize))
              .attr('cy', d => this.yPx(d[1], pSize));
          }
        }

        // -------- IMLE gradient arrows (gen → real) in gradient panels --------
        const gradDotsElementList = [
          '#vis-generator-gradients',
          '#svg-generator-gradients'
        ];
        if (this.iterationCount === 1) {
          gradDotsElementList.forEach((sel) => {
            const size = this.sizeFor(sel);
            const arrowWidth = (size === this.plotSizePx) ? 0.004 : 0.002;
            d3.select(sel)
              .selectAll('.gradient-generated')
              .data(gradData)
              .enter()
              .append('polygon')
              .attr('class', 'gradient-generated gan-lab')
              .attr('points', (d: number[]) =>
                this.createArrowPolygon(d, size, arrowWidth));
          });

          const layered = d3.select('#vis-generated-samples');
          layered.selectAll('.true-dot').lower();
          layered.selectAll('.generated-dot').raise();
          layered.selectAll('.vis-discriminator-output').raise();
          layered.selectAll('.gradient-generated').raise();
        }

        gradDotsElementList.forEach((dotsElement, k) => {
          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
          const arrowWidth = k === 0 ? 0.004 : 0.03;
          const lenScale   = k === 0 ? 5 : 3.0;  // make arrows longer in small panel

          d3Transition.transition()
            .select(dotsElement)
            .selectAll('.gradient-generated').selection().data(gradData)
            .transition().duration(SLOW_INTERVAL_MS)
            .attr('points', (d: number[]) =>
              this.createArrowPolygonScaled(d, plotSizePx, arrowWidth, lenScale));
        });
      });
    }

    if (this.slowMode) {
      await this.highlightIMLEStep(['component-g-loss', 'component-generated-prediction'], 'tooltip-imle-loss');
      document.getElementById('component-g-loss')!.classList.add('activated');
      for (let i = 0; i < this.gFlowElements.length; ++i) {
        this.gFlowElements[i].classList.add('g-activated');
      }
    }

    // --------------------------------------------
    // Train generator with IMLE (matched-noise)
    // --------------------------------------------
    const kGSteps = this.kGSteps;
    let gCostVal: number | null = null;

    tf.tidy(() => {
      for (let j = 0; j < kGSteps; j++) {
        const gCost = this.model.gOptimizer.minimize(() => {
          const realBatch = this.trueSampleProvider.getNextCopy() as tf.Tensor2D;
          const latents = this.getLatentPoolFromFixedProvider(this.sampleFactor);
          const genBatch = this.model.generator(latents);
          const nnIdx = this.model.nearest_neighbour(realBatch, genBatch);
          const matchedNoise = latents.gather(nnIdx);
          return this.model.imleLoss(realBatch, matchedNoise);
        }, true, this.model.gVariables);
        if ((!keepIterating || this.iterationCount === 1 || this.slowMode ||
            this.iterationCount % VIS_INTERVAL === 0) && j + 1 === kGSteps) {
          gCostVal = gCost.get();
        }
      }
    });

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % VIS_INTERVAL === 0) {
      // Update generator loss.
      if (gCostVal != null) {
        document.getElementById('g-loss-value')!.innerText = gCostVal.toFixed(3);
        document.getElementById('g-loss-bar')!.title = gCostVal.toFixed(3);
        const scaled = Math.log10(gCostVal + 1e-6);
        const normalized = Math.min(Math.max((scaled + 4) / 4, 0), 1);
        const barWidth = 1 + normalized * (100 - 5);
        document.getElementById('g-loss-bar')!.style.width = `${barWidth}px`;
      }

      // Charts
      if (this.iterationCount === 1) {
        const chartContainer = document.getElementById('chart-container') as HTMLElement;
        chartContainer.style.visibility = 'visible';
      }

      this.updateChartData(this.costChartData, this.iterationCount, [gCostVal]);
      this.costChart.update();

      if (this.slowMode) {
        await this.highlightIMLEStep(['component-generator-gradients'], 'tooltip-g-gradients');
        await this.highlightIMLEStep(['component-generator'], 'tooltip-update-generator');
      }

      // Manifold (noise→output) for 1D/2D latents
      tf.tidy(() => {
        if (this.noiseSize <= 2) {
          const manifoldData: Float32Array[] = [];
          const numBatches = Math.ceil(Math.pow(NUM_MANIFOLD_CELLS + 1, this.noiseSize) / BATCH_SIZE);
          const remainingDummy = BATCH_SIZE * numBatches - Math.pow(
            NUM_MANIFOLD_CELLS + 1, this.noiseSize) * this.noiseSize;
          for (let k = 0; k < numBatches; ++k) {
            const noiseBatch = this.uniformNoiseProvider.getNextCopy() as tf.Tensor2D;
            const result = this.model.generator(noiseBatch);
            const maniResult = result.dataSync() as Float32Array;
            for (let i = 0; i < (k + 1 < numBatches ? BATCH_SIZE : BATCH_SIZE - remainingDummy); ++i) {
              manifoldData.push(maniResult.slice(i * 2, i * 2 + 2));
            }
          }

          const gridData: ManifoldCell[] = this.noiseSize === 1
            ? [{ points: manifoldData }]
            : this.createGridCellsFromManifoldData(manifoldData);

          const gManifoldElementList = ['#vis-manifold', '#svg-generator-manifold'];
          gManifoldElementList.forEach((sel) => {
            const size = this.sizeFor(sel);
            const manifoldCell = line()
              .x((d: number[]) => this.xPx(d[0], size))
              .y((d: number[]) => this.yPx(d[1], size));

            if (this.iterationCount === 1) {
              d3.select(sel)
                .selectAll('.grids')
                .data(gridData)
                .enter()
                .append('g')
                .attr('class', 'grids gan-lab')
                .append('path')
                .attr('class', 'manifold-cell gan-lab')
                .style('fill', () => this.noiseSize === 2 ? '#7b3294' : 'none');
            }
            d3.select(sel)
              .selectAll('.grids')
              .data(gridData)
              .select('.manifold-cell')
              .attr('d', (d: ManifoldCell) => manifoldCell(
                d.points.map(point => [point[0], point[1]] as [number, number])
              ))
              .style('fill-opacity', (d: ManifoldCell, i: number) => {
                return this.selectedNoiseType === '2D Gaussian'
                  ? Math.min(0.1 + this.densitiesForGaussian[i] /
                    (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.2, 0.9)
                  : (this.noiseSize === 2 ? Math.max(
                    0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
                    : 'none');
              });

            if (this.noiseSize === 1) {
              const manifoldDots =
                d3.select(sel)
                  .selectAll('.uniform-generated-dot')
                  .data(manifoldData);
              if (this.iterationCount === 1) {
                manifoldDots.enter()
                  .append('circle')
                  .attr('class', 'uniform-generated-dot gan-lab')
                  .attr('r', 1);
              }
              manifoldDots
                .attr('cx', (d: Float32Array) => this.xPx(d[0], size))
                .attr('cy', (d: Float32Array) => this.yPx(d[1], size));
            }
          });
        }
      });

      // Update big generated plot post-update
      const gData: Array<[number, number]> = [];
      tf.tidy(() => {
        const noiseFixedBatch = this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gResult = this.model.generator(noiseFixedBatch);
        const gResultData = gResult.dataSync();
        for (let i = 0; i < gResultData.length / 2; ++i) {
          gData.push([gResultData[i * 2], gResultData[i * 2 + 1]]);
        }
      });

      if (!this.slowMode) {
        this.gDotsElementList.forEach((sel) => {
          const size = this.sizeFor(sel);
          d3Transition.transition()
            .select(sel)
            .selectAll('.generated-dot')
            .selection()
            .data(gData)
            .transition().duration(SLOW_INTERVAL_MS)
            .attr('cx', (d: number[]) => this.xPx(d[0], size))
            .attr('cy', (d: number[]) => this.yPx(d[1], size))
            .select('title').text((d: number[], i: number) =>
              `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)} (${i})`);
        });
      }

      // Simple grid-based evaluation
      this.evaluator.updateGridsForGenerated(gData);
      this.updateChartData(this.evalChartData, this.iterationCount, [
        this.evaluator.getKLDivergenceScore(),
        this.evaluator.getJSDivergenceScore()
      ]);
      this.evalChart.update();

      if (this.slowMode) {
        await this.sleep(SLOW_INTERVAL_MS);
        this.dehighlightStep();

        const container = document.getElementById('model-visualization-container')!;
        if (container.classList.contains('any-highlighted')) {
          container.classList.remove('any-highlighted');
        }
        document.getElementById('component-g-loss')!.classList.remove('activated');
        for (let i = 0; i < this.gFlowElements.length; ++i) {
          this.gFlowElements[i].classList.remove('g-activated');
        }
      }
    }

    if (this.iterationCount >= 999999) {
      this.isPlaying = false;
    }

    requestAnimationFrame(() => this.iterateTraining(true));
  }

  private createArrowPolygon(d: number[],
    plotSizePx: number, arrowWidth: number) {
    const gradSize = Math.sqrt(d[2] * d[2] + d[3] * d[3] + 0.00000001);
    const xNorm = d[2] / gradSize;
    const yNorm = d[3] / gradSize;
    return `${d[0] * plotSizePx},
      ${(1.0 - d[1]) * plotSizePx}
      ${(d[0] - yNorm * (-1) * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * arrowWidth)) * plotSizePx}
      ${(d[0] + d[2] * GRAD_ARROW_UNIT_LEN) * plotSizePx},
      ${(1.0 - (d[1] + d[3] * GRAD_ARROW_UNIT_LEN)) * plotSizePx}
      ${(d[0] - yNorm * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * (-1) * arrowWidth)) * plotSizePx}`;
  }

  private createArrowPolygonScaled(d: number[], plotSizePx: number, arrowWidth: number, lenScale: number) {
    const gradSize = Math.sqrt(d[2] * d[2] + d[3] * d[3] + 1e-8);
    const xNorm = d[2] / gradSize;
    const yNorm = d[3] / gradSize;
    const len = GRAD_ARROW_UNIT_LEN * lenScale;

    return `${d[0] * plotSizePx},
      ${(1.0 - d[1]) * plotSizePx}
      ${(d[0] - yNorm * (-1) * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * arrowWidth)) * plotSizePx}
      ${(d[0] + d[2] * len) * plotSizePx},
      ${(1.0 - (d[1] + d[3] * len)) * plotSizePx}
      ${(d[0] - yNorm * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * (-1) * arrowWidth)) * plotSizePx}`;
  }


  private createGridCellsFromManifoldData(manifoldData: Float32Array[]) {
    const gridData: ManifoldCell[] = [];
    let areaSum = 0.0;
    for (let i = 0; i < NUM_MANIFOLD_CELLS * NUM_MANIFOLD_CELLS; ++i) {
      const x = i % NUM_MANIFOLD_CELLS;
      const y = Math.floor(i / NUM_MANIFOLD_CELLS);
      const index = x + y * (NUM_MANIFOLD_CELLS + 1);

      const gridCell: Float32Array[] = [];
      gridCell.push(manifoldData[index]);
      gridCell.push(manifoldData[index + 1]);
      gridCell.push(manifoldData[index + 1 + (NUM_MANIFOLD_CELLS + 1)]);
      gridCell.push(manifoldData[index + (NUM_MANIFOLD_CELLS + 1)]);
      gridCell.push(manifoldData[index]);

      // polygon area
      let area = 0.0;
      for (let j = 0; j < 4; ++j) {
        area += gridCell[j % 4][0] * gridCell[(j + 1) % 4][1] -
          gridCell[j % 4][1] * gridCell[(j + 1) % 4][0];
      }
      area = 0.5 * Math.abs(area);
      areaSum += area;

      gridData.push({ points: gridCell, area });
    }
    gridData.forEach(grid => {
      if (grid.area) grid.area = grid.area / areaSum;
    });

    return gridData;
  }

  private playGeneratorAnimation() {
    if (this.noiseSize <= 2) {
      const manifoldData: Float32Array[] = [];
      const numBatches = Math.ceil(Math.pow(NUM_MANIFOLD_CELLS + 1, this.noiseSize) / BATCH_SIZE);
      const remainingDummy = BATCH_SIZE * numBatches - Math.pow(
        NUM_MANIFOLD_CELLS + 1, this.noiseSize) * 2;
      for (let k = 0; k < numBatches; ++k) {
        const maniArray: Float32Array =
          this.uniformNoiseProvider.getNextCopy().dataSync() as Float32Array;
        for (let i = 0; i < (k + 1 < numBatches ? BATCH_SIZE : BATCH_SIZE - remainingDummy); ++i) {
          if (this.noiseSize >= 2) {
            manifoldData.push(maniArray.slice(i * 2, i * 2 + 2));
          } else {
            manifoldData.push(new Float32Array([maniArray[i], 0.5]));
          }
        }
      }

      const noiseData = this.noiseSize === 1
        ? [{ points: manifoldData }]
        : this.createGridCellsFromManifoldData(manifoldData);

      const gridData = d3.select('#svg-generator-manifold')
        .selectAll('.grids').data();

      const uniformDotsData = d3.select('#svg-generator-manifold')
        .selectAll('.uniform-generated-dot').data();

      const size = this.sizeFor('#svg-generator-manifold');
      const manifoldCell = line()
        .x((d: number[]) => this.xPx(d[0], size))
        .y((d: number[]) => this.yPx(d[1], size));

      // Visualize noise.
      d3.select('#svg-generator-manifold')
        .selectAll('.grids')
        .data(noiseData)
        .select('.manifold-cell')
        .attr('d', (d: ManifoldCell) => manifoldCell(
          d.points.map(point => [point[0], point[1]] as [number, number])
        ))
        .style('fill-opacity', (d: ManifoldCell, i: number) => {
          return this.selectedNoiseType === '2D Gaussian'
            ? Math.min(0.1 + this.densitiesForGaussian[i] /
              (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.2, 0.9)
            : (this.noiseSize === 2 ? Math.max(
              0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
              : 'none');
        });

      if (this.noiseSize === 1) {
        d3.select('#svg-generator-manifold')
          .selectAll('.uniform-generated-dot')
          .data(manifoldData)
          .attr('cx', (d: Float32Array) => this.xPx(d[0], size))
          .attr('cy', (d: Float32Array) => this.yPx(d[1], size));
      }

      // Transition to current manifold
      d3Transition.transition()
        .select('#svg-generator-manifold')
        .selectAll('.grids')
        .selection()
        .data(gridData)
        .transition().duration(2000)
        .select('.manifold-cell')
        .attr('d', (d: ManifoldCell) => manifoldCell(
          d.points.map(point => [point[0], point[1]] as [number, number])
        ))
        .style('fill-opacity', (d: ManifoldCell, i: number) => {
          return this.selectedNoiseType === '2D Gaussian'
            ? Math.min(0.1 + this.densitiesForGaussian[i] /
              (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.3, 0.9)
            : (this.noiseSize === 2 ? Math.max(
              0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
              : 'none');
        });

      if (this.noiseSize === 1) {
        d3Transition.transition()
          .select('#svg-generator-manifold')
          .selectAll('.uniform-generated-dot')
          .selection()
          .data(uniformDotsData)
          .transition().duration(2000)
          .attr('cx', (d: Float32Array) => this.xPx(d[0], size))
          .attr('cy', (d: Float32Array) => this.yPx(d[1], size));
      }
    }
  }
private async highlightIMLEStep(
  componentElementNames: string[],
  tooltipElementName: string
): Promise<void> {
  await this.sleep(SLOW_INTERVAL_MS);
  this.dehighlightStep();

  // Safely collect valid DOM elements
  this.highlightedComponents = componentElementNames
    .map(name => document.getElementById(name) as HTMLDivElement | null)
    .filter((el): el is HTMLDivElement => el !== null);

  this.highlightedTooltip = document.getElementById(tooltipElementName) as HTMLDivElement | null;

  // Add highlight classes
  this.highlightedComponents.forEach(component => {
    component.classList.add('highlighted');
  });

  if (this.highlightedTooltip) {
    this.highlightedTooltip.classList.add('shown');
    this.highlightedTooltip.classList.add('highlighted');
  }

  await this.sleep(SLOW_INTERVAL_MS);
}

private dehighlightStep(): void {
  if (this.highlightedComponents) {
    this.highlightedComponents.forEach(component => {
      component.classList.remove('highlighted');
    });
  }
  if (this.highlightedTooltip) {
    this.highlightedTooltip.classList.remove('shown');
    this.highlightedTooltip.classList.remove('highlighted');
  }
}


  private async onClickSaveModelButton() {
    const gTensors: tf.NamedTensorMap =
      this.model.gVariables.reduce((obj, item, i) => {
        obj[`g-${i}`] = item;
        return obj;
      }, {} as tf.NamedTensorMap);
    const tensors: tf.NamedTensorMap = { ...gTensors };

    const modelInfo: {} = {
      'shape_name': this.selectedShapeName,
      'iter_count': this.iterationCount,
      'config': {
        selectedNoiseType: this.selectedNoiseType,
        noiseSize: this.noiseSize,
        numGeneratorLayers: this.numGeneratorLayers,
        numDiscriminatorLayers: this.numDiscriminatorLayers,
        numGeneratorNeurons: this.numGeneratorNeurons,
        numDiscriminatorNeurons: this.numDiscriminatorNeurons,
        dLearningRate: this.dLearningRate,
        gLearningRate: this.gLearningRate,
        gOptimizerType: this.gOptimizerType,
        distanceType: this.distanceType,
        kGSteps: this.kGSteps,
      }
    };
    const weightDataAndSpecs = await tf.io.encodeWeights(tensors);
    const modelArtifacts: tf.io.ModelArtifacts = {
      modelTopology: modelInfo,
      weightSpecs: weightDataAndSpecs.specs,
      weightData: weightDataAndSpecs.data,
    };

    const downloadTrigger =
      tf.io.getSaveHandlers('downloads://ganlab_trained_model')[0];
    await downloadTrigger.save(modelArtifacts);
  }

  private async loadPretrainedWeightFile(filename: string):
      Promise<tf.io.ModelArtifacts> {
    const handler =
      tf.io.browserHTTPRequest(`pretrained_models/${filename}.json`);
    const loadedModel: tf.io.ModelArtifacts = await handler.load();

    this.iterationCount = (loadedModel.modelTopology as any)['iter_count'];

    const loadedConfig: any = (loadedModel.modelTopology as any)['config'];
    for (const configProperty in loadedConfig) {
      (this as any)[configProperty] = loadedConfig[configProperty];
    }

    document.getElementById('num-g-layers')!.innerText =
      this.numGeneratorLayers.toString();
    document.getElementById('num-g-neurons')!.innerText =
      this.numGeneratorNeurons.toString();
    document.getElementById('k-g-steps')!.innerText = this.kGSteps.toString();

    return loadedModel as Promise<tf.io.ModelArtifacts>;
  }

  private recreateCharts() {
    document.getElementById('chart-container')!.style.visibility = 'hidden';

    this.costChartData = new Array<ChartData>(1);
    for (let i = 0; i < this.costChartData.length; ++i) this.costChartData[i] = [];
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    const costChartSpecification = [
      { label: 'Generator\'s Loss', color: 'rgba(123, 50, 148, 0.5)' }
    ];
    this.costChart = this.createChart(
      'cost-chart', this.costChartData, costChartSpecification, 0);

    this.evalChartData = new Array<ChartData>(2);
    for (let i = 0; i < this.evalChartData.length; ++i) this.evalChartData[i] = [];
    if (this.evalChart != null) {
      this.evalChart.destroy();
    }
    const evalChartSpecification = [
      { label: 'KL Divergence (by grid)', color: 'rgba(220, 80, 20, 0.5)' },
      { label: 'JS Divergence (by grid)', color: 'rgba(200, 150, 10, 0.5)' }
    ];
    this.evalChart = this.createChart(
      'eval-chart', this.evalChartData, evalChartSpecification, 0);
  }

  private updateChartData(data: ChartData[], xVal: number, yList: Array<number | null>) {
    for (let i = 0; i < yList.length; ++i) {
      const y = yList[i];
      data[i].push({ x: xVal, y: y == null ? null : +y.toFixed(3) });
    }
  }

  private createChart(
    canvasId: string, chartData: ChartData[],
    specification: Array<{ label: string, color: string }>,
    min?: number, max?: number): any {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
      .getContext('2d') as CanvasRenderingContext2D;
    const chartDatasets = specification.map((chartSpec, i) => {
      return {
        data: chartData[i],
        backgroundColor: chartSpec.color,
        borderColor: chartSpec.color,
        borderWidth: 1,
        fill: false,
        label: chartSpec.label,
        lineTension: 0,
        pointHitRadius: 8,
        pointRadius: 0
      };
    });

    return new Chart(context, {
      type: 'line',
      data: { datasets: chartDatasets },
      options: {
        animation: { duration: 0 },
        legend: {
          labels: { boxWidth: 10 }
        },
        responsive: false,
        scales: {
          xAxes: [{ type: 'linear', position: 'bottom' }],
          yAxes: [{ ticks: { max, min } }]
        }
      }
    });
  }

  private sleep(ms: number) {
    return new Promise(resolve => {
      const check = () => {
        if (this.isPlaying) {
          this.isPausedOngoingIteration = false;
          resolve(null);
        } else {
          this.isPausedOngoingIteration = true;
          setTimeout(check, 1000);
        }
      };
      setTimeout(check, ms);
    });
  }
}

document.registerElement(GANLab.prototype.is, GANLab);
