import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import ProbabilityStatisticsControls from '../components/probabilityStatistics/ProbabilityStatisticsControls';
import ProbabilityStatisticsVisualization from '../components/probabilityStatistics/ProbabilityStatisticsVisualization';
import ProbabilityStatisticsEducationalPanels from '../components/probabilityStatistics/ProbabilityStatisticsEducationalPanels';

export default function ProbabilityStatistics() {
  const [selectedTopic, setSelectedTopic] = useState('descriptive');
  const [dataSet, setDataSet] = useState([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-600">
            Probability & Statistics for AI & ML
          </h1>
          <p className="text-gray-700 mt-2">
            Master descriptive statistics, probability distributions, Bayes' theorem, and their applications in ML
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <ProbabilityStatisticsControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
              dataSet={dataSet}
              setDataSet={setDataSet}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <ProbabilityStatisticsVisualization
              selectedTopic={selectedTopic}
              dataSet={dataSet}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <ProbabilityStatisticsEducationalPanels
            selectedTopic={selectedTopic}
            dataSet={dataSet}
          />
        </div>
      </div>
    </div>
  );
}

