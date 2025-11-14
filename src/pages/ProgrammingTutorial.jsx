import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import ProgrammingTutorialControls from '../components/programmingTutorial/ProgrammingTutorialControls';
import ProgrammingTutorialVisualization from '../components/programmingTutorial/ProgrammingTutorialVisualization';
import ProgrammingTutorialEducationalPanels from '../components/programmingTutorial/ProgrammingTutorialEducationalPanels';

export default function ProgrammingTutorial() {
  const [selectedTopic, setSelectedTopic] = useState('pytorch-basics');
  const [selectedFramework, setSelectedFramework] = useState('pytorch');

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      {/* Header */}
      <div className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link
            to="/"
            className="inline-flex items-center text-indigo-600 hover:text-indigo-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Home
          </Link>
          <h1 className="text-4xl font-bold text-gray-900">
            Programming Tutorial
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Hands-on programming exercises with PyTorch and TensorFlow. Build models from scratch and use pre-trained models.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1">
            <ProgrammingTutorialControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
              selectedFramework={selectedFramework}
              setSelectedFramework={setSelectedFramework}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <ProgrammingTutorialVisualization
              selectedTopic={selectedTopic}
              selectedFramework={selectedFramework}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <ProgrammingTutorialEducationalPanels
            selectedTopic={selectedTopic}
            selectedFramework={selectedFramework}
          />
        </div>
      </div>
    </div>
  );
}

