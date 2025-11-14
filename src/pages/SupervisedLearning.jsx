import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import SupervisedLearningControls from '../components/supervisedLearning/SupervisedLearningControls';
import SupervisedLearningVisualization from '../components/supervisedLearning/SupervisedLearningVisualization';
import SupervisedLearningEducationalPanels from '../components/supervisedLearning/SupervisedLearningEducationalPanels';

export default function SupervisedLearning() {
  const [selectedTopic, setSelectedTopic] = useState('foundations');

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-green-600 hover:text-green-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-emerald-600">
            Supervised Learning
          </h1>
          <p className="text-gray-700 mt-2">
            Learn loss functions, model evaluation, bias-variance tradeoff, and regularization techniques
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <SupervisedLearningControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <SupervisedLearningVisualization
              selectedTopic={selectedTopic}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <SupervisedLearningEducationalPanels
            selectedTopic={selectedTopic}
          />
        </div>
      </div>
    </div>
  );
}

