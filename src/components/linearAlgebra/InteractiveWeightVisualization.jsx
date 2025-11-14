import React, { useState } from 'react';
import { Zap, Eye, EyeOff } from 'lucide-react';

export default function InteractiveWeightVisualization() {
  const [hoveredWeight, setHoveredWeight] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [showValues, setShowValues] = useState(true);

  // Example: 3-layer network: 4 inputs -> 3 hidden -> 2 outputs
  const layers = [
    { name: 'Input Layer', size: 4, neurons: ['x₁', 'x₂', 'x₃', 'x₄'] },
    { name: 'Hidden Layer', size: 3, neurons: ['h₁', 'h₂', 'h₃'] },
    { name: 'Output Layer', size: 2, neurons: ['y₁', 'y₂'] },
  ];

  // Weight matrices between layers
  const weightMatrices = [
    [
      [0.5, -0.3, 0.8, 0.2],
      [-0.1, 0.6, -0.4, 0.9],
      [0.7, 0.1, -0.2, 0.5],
    ],
    [
      [0.3, -0.5, 0.6],
      [-0.2, 0.4, -0.8],
    ],
  ];

  const getWeightValue = (layerIdx, fromNeuron, toNeuron) => {
    return weightMatrices[layerIdx][toNeuron][fromNeuron];
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg p-6 border-2 border-purple-200">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-bold text-purple-900">Interactive Neural Network Visualization</h4>
          <button
            onClick={() => setShowValues(!showValues)}
            className="flex items-center gap-2 px-3 py-1 bg-white border-2 border-purple-300 rounded-lg hover:bg-purple-50 transition-colors"
          >
            {showValues ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            <span className="text-sm">{showValues ? 'Hide' : 'Show'} Values</span>
          </button>
        </div>

        {/* Layer Selection */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Select Weight Matrix:
          </label>
          <div className="flex gap-2">
            {weightMatrices.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedLayer(idx)}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                  selectedLayer === idx
                    ? 'bg-purple-600 text-white ring-2 ring-purple-800'
                    : 'bg-white text-purple-600 border-2 border-purple-300 hover:bg-purple-50'
                }`}
              >
                W{idx + 1}: {layers[idx].name} → {layers[idx + 1].name}
              </button>
            ))}
          </div>
        </div>

        {/* Network Visualization */}
        <div className="bg-white rounded-lg p-6 border-2 border-purple-300">
          <div className="flex items-center justify-center gap-8 overflow-x-auto pb-4">
            {/* Input Layer */}
            <div className="flex flex-col gap-3">
              <div className="text-xs font-semibold text-gray-700 text-center mb-2">
                {layers[0].name}
              </div>
              {layers[0].neurons.map((neuron, i) => (
                <div
                  key={i}
                  className="w-16 h-16 bg-blue-100 border-2 border-blue-400 rounded-full flex items-center justify-center font-semibold text-blue-900 text-sm"
                >
                  {neuron}
                </div>
              ))}
            </div>

            {/* Connections and Hidden Layer */}
            <div className="flex flex-col gap-3">
              <div className="text-xs font-semibold text-gray-700 text-center mb-2">
                {layers[1].name}
              </div>
              {layers[1].neurons.map((neuron, i) => (
                <div
                  key={i}
                  className="w-16 h-16 bg-purple-100 border-2 border-purple-400 rounded-full flex items-center justify-center font-semibold text-purple-900 text-sm relative"
                >
                  {neuron}
                  {/* Weight connections */}
                  {selectedLayer === 0 && (
                    <div className="absolute -left-20 flex flex-col gap-1">
                      {layers[0].neurons.map((_, j) => {
                        const weight = getWeightValue(0, j, i);
                        const weightKey = `0-${j}-${i}`;
                        const isHovered = hoveredWeight === weightKey;
                        return (
                          <div
                            key={j}
                            className="relative"
                            onMouseEnter={() => setHoveredWeight(weightKey)}
                            onMouseLeave={() => setHoveredWeight(null)}
                          >
                            <div
                              className={`w-16 h-0.5 transition-all ${
                                isHovered
                                  ? 'bg-yellow-500 h-1'
                                  : weight > 0
                                  ? 'bg-green-500'
                                  : 'bg-red-500'
                              }`}
                            />
                            {showValues && (
                              <div
                                className={`absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-mono px-1 rounded ${
                                  isHovered
                                    ? 'bg-yellow-200 text-gray-900 font-bold'
                                    : 'bg-white text-gray-600'
                                }`}
                              >
                                {weight.toFixed(2)}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Output Layer */}
            <div className="flex flex-col gap-3">
              <div className="text-xs font-semibold text-gray-700 text-center mb-2">
                {layers[2].name}
              </div>
              {layers[2].neurons.map((neuron, i) => (
                <div
                  key={i}
                  className="w-16 h-16 bg-green-100 border-2 border-green-400 rounded-full flex items-center justify-center font-semibold text-green-900 text-sm relative"
                >
                  {neuron}
                  {/* Weight connections */}
                  {selectedLayer === 1 && (
                    <div className="absolute -left-20 flex flex-col gap-1">
                      {layers[1].neurons.map((_, j) => {
                        const weight = getWeightValue(1, j, i);
                        const weightKey = `1-${j}-${i}`;
                        const isHovered = hoveredWeight === weightKey;
                        return (
                          <div
                            key={j}
                            className="relative"
                            onMouseEnter={() => setHoveredWeight(weightKey)}
                            onMouseLeave={() => setHoveredWeight(null)}
                          >
                            <div
                              className={`w-16 h-0.5 transition-all ${
                                isHovered
                                  ? 'bg-yellow-500 h-1'
                                  : weight > 0
                                  ? 'bg-green-500'
                                  : 'bg-red-500'
                              }`}
                            />
                            {showValues && (
                              <div
                                className={`absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-mono px-1 rounded ${
                                  isHovered
                                    ? 'bg-yellow-200 text-gray-900 font-bold'
                                    : 'bg-white text-gray-600'
                                }`}
                              >
                                {weight.toFixed(2)}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Weight Matrix Display */}
          <div className="mt-6 bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h5 className="font-semibold text-gray-900 mb-3">
              Weight Matrix W{selectedLayer + 1} ({layers[selectedLayer + 1].size}×{layers[selectedLayer].size})
            </h5>
            <div className="bg-white rounded p-3 border-2 border-purple-300">
              <div className="font-mono text-sm">
                {weightMatrices[selectedLayer].map((row, i) => (
                  <div key={i} className="flex gap-2 justify-center mb-1">
                    {row.map((val, j) => {
                      const weightKey = `${selectedLayer}-${j}-${i}`;
                      const isHovered = hoveredWeight === weightKey;
                      return (
                        <div
                          key={j}
                          onMouseEnter={() => setHoveredWeight(weightKey)}
                          onMouseLeave={() => setHoveredWeight(null)}
                          className={`w-16 text-center px-2 py-1 rounded transition-all cursor-pointer ${
                            isHovered
                              ? 'bg-yellow-400 text-gray-900 font-bold ring-2 ring-yellow-600 scale-110'
                              : val > 0
                              ? 'bg-green-100 text-green-900'
                              : 'bg-red-100 text-red-900'
                          }`}
                        >
                          {val.toFixed(2)}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="mt-2 flex items-center justify-center gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-4 h-4 bg-green-100 border border-green-300"></div>
                <span>Positive weight</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-4 h-4 bg-red-100 border border-red-300"></div>
                <span>Negative weight</span>
              </div>
            </div>
          </div>

          {/* Hover Info */}
          {hoveredWeight && (
            <div className="mt-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg p-3">
              <div className="text-sm text-yellow-900">
                <strong>Weight:</strong> {getWeightValue(
                  parseInt(hoveredWeight.split('-')[0]),
                  parseInt(hoveredWeight.split('-')[1]),
                  parseInt(hoveredWeight.split('-')[2])
                ).toFixed(4)}
              </div>
              <div className="text-xs text-yellow-800 mt-1">
                Connects {layers[parseInt(hoveredWeight.split('-')[0])].neurons[parseInt(hoveredWeight.split('-')[1])]} 
                {' → '}
                {layers[parseInt(hoveredWeight.split('-')[0]) + 1].neurons[parseInt(hoveredWeight.split('-')[2])]}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

