import React from 'react';
import PyTorchBasics from './PyTorchBasics';
import TensorFlowBasics from './TensorFlowBasics';
import LinearRegression from './LinearRegression';
import LogisticRegression from './LogisticRegression';
import NeuralNetwork from './NeuralNetwork';
import PretrainedModels from './PretrainedModels';
import DataLoading from './DataLoading';
import TrainingLoops from './TrainingLoops';
import ModelEvaluation from './ModelEvaluation';

export default function ProgrammingTutorialVisualization({ selectedTopic, selectedFramework }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Code & Output</h2>

      {selectedTopic === 'pytorch-basics' && (
        <PyTorchBasics />
      )}

      {selectedTopic === 'tensorflow-basics' && (
        <TensorFlowBasics />
      )}

      {selectedTopic === 'linear-regression' && (
        <LinearRegression framework={selectedFramework} />
      )}

      {selectedTopic === 'logistic-regression' && (
        <LogisticRegression framework={selectedFramework} />
      )}

      {selectedTopic === 'neural-network' && (
        <NeuralNetwork framework={selectedFramework} />
      )}

      {selectedTopic === 'pretrained-models' && (
        <PretrainedModels framework={selectedFramework} />
      )}

      {selectedTopic === 'data-loading' && (
        <DataLoading framework={selectedFramework} />
      )}

      {selectedTopic === 'training-loops' && (
        <TrainingLoops framework={selectedFramework} />
      )}

      {selectedTopic === 'model-evaluation' && (
        <ModelEvaluation framework={selectedFramework} />
      )}
    </div>
  );
}

