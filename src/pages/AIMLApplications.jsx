import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import AIMLApplicationsControls from '../components/aimlApplications/AIMLApplicationsControls';
import AIMLApplicationsVisualization from '../components/aimlApplications/AIMLApplicationsVisualization';
import AIMLApplicationsEducationalPanels from '../components/aimlApplications/AIMLApplicationsEducationalPanels';

export default function AIMLApplications() {
  const [selectedApplication, setSelectedApplication] = useState('image-classification');

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
            Real-World AI/ML Applications
          </h1>
          <p className="mt-2 text-lg text-gray-600">
            Complete end-to-end tutorials building real AI/ML applications. Put all concepts together into practical projects.
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls */}
          <div className="lg:col-span-1">
            <AIMLApplicationsControls
              selectedApplication={selectedApplication}
              setSelectedApplication={setSelectedApplication}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <AIMLApplicationsVisualization
              selectedApplication={selectedApplication}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <AIMLApplicationsEducationalPanels
            selectedApplication={selectedApplication}
          />
        </div>
      </div>
    </div>
  );
}

