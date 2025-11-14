import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import UnsupervisedLearningControls from '../components/unsupervisedLearning/UnsupervisedLearningControls';
import UnsupervisedLearningVisualization from '../components/unsupervisedLearning/UnsupervisedLearningVisualization';
import UnsupervisedLearningEducationalPanels from '../components/unsupervisedLearning/UnsupervisedLearningEducationalPanels';

export default function UnsupervisedLearning() {
  const [selectedTopic, setSelectedTopic] = useState('clustering');

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-cyan-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-teal-600 hover:text-teal-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-600">
            Unsupervised Learning
          </h1>
          <p className="text-gray-700 mt-2">
            Learn clustering, dimensionality reduction, anomaly detection, and distance metrics for finding patterns in unlabeled data
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <UnsupervisedLearningControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <UnsupervisedLearningVisualization
              selectedTopic={selectedTopic}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <UnsupervisedLearningEducationalPanels
            selectedTopic={selectedTopic}
          />
        </div>
      </div>
    </div>
  );
}

