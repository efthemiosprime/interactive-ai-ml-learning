import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import CalculusControls from '../components/calculus/CalculusControls';
import CalculusVisualization from '../components/calculus/CalculusVisualization';
import CalculusEducationalPanels from '../components/calculus/CalculusEducationalPanels';

export default function Calculus() {
  const [selectedTopic, setSelectedTopic] = useState('derivatives');
  const [functionType, setFunctionType] = useState('quadratic');

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">
            Calculus for AI & ML
          </h1>
          <p className="text-gray-700 mt-2">
            Learn derivatives, gradients, chain rule, and how they power backpropagation in neural networks
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <CalculusControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
              functionType={functionType}
              setFunctionType={setFunctionType}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <CalculusVisualization
              selectedTopic={selectedTopic}
              functionType={functionType}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <CalculusEducationalPanels
            selectedTopic={selectedTopic}
            functionType={functionType}
          />
        </div>
      </div>
    </div>
  );
}

