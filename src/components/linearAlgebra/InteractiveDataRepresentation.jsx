import React, { useState } from 'react';
import { Info, X } from 'lucide-react';

export default function InteractiveDataRepresentation() {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);

  // Example data matrix: 4 samples, 3 features
  const dataMatrix = [
    [25, 50000, 2],   // Sample 1: Age, Income, Purchases
    [35, 75000, 5],   // Sample 2
    [28, 60000, 3],   // Sample 3
    [45, 90000, 8],   // Sample 4
  ];

  const features = ['Age', 'Income ($)', 'Purchases'];
  const samples = ['Customer 1', 'Customer 2', 'Customer 3', 'Customer 4'];

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
        <h4 className="font-bold text-indigo-900 mb-4">Interactive Data Matrix</h4>
        <p className="text-sm text-gray-700 mb-4">
          Hover over cells to see details. Click a row to highlight the entire sample.
        </p>

        {/* Interactive Matrix */}
        <div className="bg-white rounded-lg p-4 border-2 border-indigo-300 overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Header Row */}
            <div className="flex gap-2 mb-2">
              <div className="w-24 flex-shrink-0"></div>
              {features.map((feature, idx) => (
                <div
                  key={idx}
                  className="w-24 text-center px-2 py-1 bg-indigo-100 text-indigo-900 font-semibold rounded text-sm"
                >
                  {feature}
                </div>
              ))}
            </div>

            {/* Data Rows */}
            {dataMatrix.map((row, i) => (
              <div
                key={i}
                className={`flex gap-2 mb-1 transition-all cursor-pointer ${
                  selectedSample === i ? 'bg-yellow-100 rounded px-2 py-1' : ''
                }`}
                onClick={() => setSelectedSample(selectedSample === i ? null : i)}
              >
                <div className="w-24 flex-shrink-0 text-sm font-semibold text-gray-700 flex items-center">
                  {samples[i]}
                </div>
                {row.map((val, j) => {
                  const cellKey = `${i}-${j}`;
                  const isHovered = hoveredCell === cellKey;
                  return (
                    <div
                      key={j}
                      onMouseEnter={() => setHoveredCell(cellKey)}
                      onMouseLeave={() => setHoveredCell(null)}
                      className={`w-24 text-center px-2 py-2 rounded transition-all cursor-pointer ${
                        isHovered
                          ? 'bg-indigo-600 text-white font-bold scale-110 ring-2 ring-yellow-400 z-10 relative'
                          : selectedSample === i
                          ? 'bg-yellow-200 text-gray-900 font-semibold'
                          : 'bg-gray-50 text-gray-700 hover:bg-indigo-100'
                      }`}
                    >
                      {val.toLocaleString()}
                      {isHovered && (
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap z-20">
                          Sample {i + 1}, Feature {j + 1}: {val}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Selected Sample Info */}
        {selectedSample !== null && (
          <div className="mt-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <h5 className="font-bold text-yellow-900 mb-2">
                  {samples[selectedSample]} - Sample Details
                </h5>
                <div className="space-y-1 text-sm text-yellow-800">
                  {features.map((feature, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="font-semibold">{feature}:</span>
                      <span>{dataMatrix[selectedSample][idx].toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              <button
                onClick={() => setSelectedSample(null)}
                className="text-yellow-700 hover:text-yellow-900"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* ML Context */}
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-blue-900">
              <strong>In ML:</strong> This matrix would be fed to a model. Each row (sample) contains 
              features that the model uses to make predictions. The model learns patterns across all samples.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

