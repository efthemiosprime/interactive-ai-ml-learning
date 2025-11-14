import React, { useRef, useEffect, useState } from 'react';

export default function InteractiveArchitectureVisualization() {
  const canvasRef = useRef(null);
  const [numLayers, setNumLayers] = useState(3);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState([3, 4, 2]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 800;
    const displayHeight = rect.height > 0 ? rect.height : 500;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const padding = 60;
    const layerSpacing = (width - 2 * padding) / (numLayers + 1);

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw layers
    const layers = [neuronsPerLayer[0], ...neuronsPerLayer.slice(1)];
    const neuronPositions = [];

    layers.forEach((numNeurons, layerIdx) => {
      const x = padding + layerSpacing * (layerIdx + 1);
      const neuronSpacing = (height - 2 * padding) / (numNeurons + 1);
      const layerNeurons = [];

      for (let i = 0; i < numNeurons; i++) {
        const y = padding + neuronSpacing * (i + 1);
        layerNeurons.push({ x, y });
      }

      neuronPositions.push(layerNeurons);
    });

    // Draw connections
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let l = 0; l < neuronPositions.length - 1; l++) {
      const currentLayer = neuronPositions[l];
      const nextLayer = neuronPositions[l + 1];

      currentLayer.forEach(neuron1 => {
        nextLayer.forEach(neuron2 => {
          ctx.beginPath();
          ctx.moveTo(neuron1.x, neuron1.y);
          ctx.lineTo(neuron2.x, neuron2.y);
          ctx.stroke();
        });
      });
    }

    // Draw neurons
    neuronPositions.forEach((layer, layerIdx) => {
      layer.forEach((neuron, neuronIdx) => {
        // Input layer
        if (layerIdx === 0) {
          ctx.fillStyle = '#3b82f6';
        }
        // Output layer
        else if (layerIdx === neuronPositions.length - 1) {
          ctx.fillStyle = '#10b981';
        }
        // Hidden layers
        else {
          ctx.fillStyle = '#8b5cf6';
        }

        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, 15, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    });

    // Draw labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';

    // Layer labels
    const layerLabels = ['Input', ...Array(numLayers - 2).fill('Hidden'), 'Output'];
    neuronPositions.forEach((layer, idx) => {
      ctx.fillText(layerLabels[idx], layer[0].x, layer[0].y - 30);
      ctx.fillText(`${layer.length} neurons`, layer[0].x, layer[0].y - 15);
    });

    // Title
    ctx.fillText('Neural Network Architecture', width / 2, 25);
  }, [numLayers, neuronsPerLayer]);

  const updateNeurons = (layerIdx, value) => {
    const newNeurons = [...neuronsPerLayer];
    newNeurons[layerIdx] = parseInt(value) || 1;
    setNeuronsPerLayer(newNeurons);
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> Adjust the network architecture to see how layers and neurons are connected!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Number of Layers: {numLayers}
          </label>
          <input
            type="range"
            min="2"
            max="5"
            step="1"
            value={numLayers}
            onChange={(e) => {
              const newNum = parseInt(e.target.value);
              setNumLayers(newNum);
              const newNeurons = [];
              for (let i = 0; i < newNum; i++) {
                newNeurons.push(neuronsPerLayer[i] || 3);
              }
              setNeuronsPerLayer(newNeurons);
            }}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Neurons per Layer:
          </label>
          {Array.from({ length: numLayers }).map((_, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <label className="text-xs text-gray-600 w-20">
                Layer {idx + 1}:
              </label>
              <input
                type="range"
                min="1"
                max="8"
                step="1"
                value={neuronsPerLayer[idx] || 3}
                onChange={(e) => updateNeurons(idx, e.target.value)}
                className="flex-1"
              />
              <span className="text-xs text-gray-600 w-8">
                {neuronsPerLayer[idx] || 3}
              </span>
            </div>
          ))}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Key Components:</h4>
        <ul className="text-sm text-gray-700 space-y-1">
          <li><strong>Input Layer:</strong> Receives input data (blue nodes)</li>
          <li><strong>Hidden Layers:</strong> Process information through weighted connections (purple nodes)</li>
          <li><strong>Output Layer:</strong> Produces final predictions (green nodes)</li>
          <li><strong>Connections:</strong> Each connection has a weight that gets adjusted during training</li>
        </ul>
      </div>
    </div>
  );
}

