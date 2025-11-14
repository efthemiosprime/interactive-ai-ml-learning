import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import NeuralNetworksControls from '../components/neuralNetworks/NeuralNetworksControls';
import NeuralNetworksVisualization from '../components/neuralNetworks/NeuralNetworksVisualization';
import NeuralNetworksEducationalPanels from '../components/neuralNetworks/NeuralNetworksEducationalPanels';

export default function NeuralNetworks() {
  const [selectedTopic, setSelectedTopic] = useState('architecture');

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-purple-50 to-fuchsia-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-violet-600 hover:text-violet-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-600 to-fuchsia-600">
            Neural Networks & Deep Learning
          </h1>
          <p className="text-gray-700 mt-2">
            Learn how neural networks work, from basic architecture to transformers and attention mechanisms that power modern LLMs
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <NeuralNetworksControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <NeuralNetworksVisualization
              selectedTopic={selectedTopic}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <NeuralNetworksEducationalPanels
            selectedTopic={selectedTopic}
          />
        </div>
      </div>
    </div>
  );
}

