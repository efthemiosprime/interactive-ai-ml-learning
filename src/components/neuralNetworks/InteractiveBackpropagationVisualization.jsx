import React, { useRef, useEffect, useState } from 'react';
import { Play } from 'lucide-react';
import * as nn from '../../utils/neuralNetworks';

export default function InteractiveBackpropagationVisualization() {
  const canvasRef = useRef(null);
  const [input, setInput] = useState([0.5, -0.3]);
  const [target, setTarget] = useState([0.8]);
  const [network, setNetwork] = useState(() => 
    nn.createNetwork([2, 3, 1], ['relu', 'sigmoid'])
  );
  const [gradients, setGradients] = useState(null);
  const [isAnimating, setIsAnimating] = useState(false);

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
    const layerSpacing = (width - 2 * padding) / (network.length + 1);

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Forward pass
    const forwardResult = nn.forwardPassNetwork(input, network);
    const activations = forwardResult.activations;

    // Backpropagation
    if (gradients) {
      const neuronPositions = [];
      activations.forEach((layerActivations, layerIdx) => {
        const x = padding + layerSpacing * layerIdx;
        const numNeurons = layerActivations.length;
        const neuronSpacing = (height - 2 * padding) / (numNeurons + 1);
        const layerNeurons = [];

        for (let i = 0; i < numNeurons; i++) {
          const y = padding + neuronSpacing * (i + 1);
          layerNeurons.push({ x, y });
        }

        neuronPositions.push(layerNeurons);
      });

      // Draw gradients flowing backward
      for (let l = neuronPositions.length - 1; l > 0; l--) {
        const currentLayer = neuronPositions[l];
        const prevLayer = neuronPositions[l - 1];
        const layerGrads = gradients[l - 1];

        currentLayer.forEach((neuron2, j) => {
          prevLayer.forEach((neuron1, i) => {
            const grad = layerGrads.weights[j][i];
            const intensity = Math.abs(grad);
            const color = grad > 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(59, 130, 246, 0.6)';
            
            ctx.strokeStyle = color;
            ctx.lineWidth = intensity * 10 + 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(neuron2.x, neuron2.y);
            ctx.lineTo(neuron1.x, neuron1.y);
            ctx.stroke();
            ctx.setLineDash([]);
          });
        });
      }

      // Draw neurons
      neuronPositions.forEach((layer, layerIdx) => {
        layer.forEach((neuron) => {
          ctx.fillStyle = '#f3f4f6';
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, 15, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#1f2937';
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      });

      // Draw gradient values
      ctx.fillStyle = '#1f2937';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      neuronPositions.forEach((layer, layerIdx) => {
        if (layerIdx < neuronPositions.length - 1 && gradients[layerIdx]) {
          layer.forEach((neuron, neuronIdx) => {
            const biasGrad = gradients[layerIdx].biases[neuronIdx];
            ctx.fillText(biasGrad.toFixed(2), neuron.x, neuron.y + 3);
          });
        }
      });
    } else {
      // Draw network without gradients
      const neuronPositions = [];
      activations.forEach((layerActivations, layerIdx) => {
        const x = padding + layerSpacing * layerIdx;
        const numNeurons = layerActivations.length;
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
      neuronPositions.forEach((layer) => {
        layer.forEach((neuron) => {
          ctx.fillStyle = '#f3f4f6';
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, 15, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#1f2937';
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      });
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Backpropagation: Gradient Flow', width / 2, 25);
  }, [input, target, network, gradients]);

  const runBackpropagation = () => {
    setIsAnimating(true);
    
    setTimeout(() => {
      const forwardResult = nn.forwardPassNetwork(input, network);
      const grads = nn.backpropagate(network, forwardResult.activations, target, 'mse');
      setGradients(grads);
      setIsAnimating(false);
    }, 100);
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> See how gradients flow backward through the network to update weights!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Input: [{input.map(v => v.toFixed(1)).join(', ')}]
          </label>
          <div className="grid grid-cols-2 gap-2">
            {input.map((val, idx) => (
              <input
                key={idx}
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={val}
                onChange={(e) => {
                  const newInput = [...input];
                  newInput[idx] = parseFloat(e.target.value);
                  setInput(newInput);
                }}
                className="w-full"
              />
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Target: [{target.map(v => v.toFixed(1)).join(', ')}]
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={target[0]}
            onChange={(e) => setTarget([parseFloat(e.target.value)])}
            className="w-full"
          />
        </div>

        <button
          onClick={runBackpropagation}
          disabled={isAnimating}
          className="w-full px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Play className="w-4 h-4" />
          Run Backpropagation
        </button>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Backpropagation Steps:</h4>
        <ol className="text-sm text-gray-700 space-y-1 list-decimal list-inside">
          <li>Calculate error at output layer</li>
          <li>Propagate error backward through each layer</li>
          <li>Calculate gradients using chain rule</li>
          <li>Update weights: w = w - α × gradient</li>
          <li>Repeat until loss is minimized</li>
        </ol>
        <p className="text-xs text-gray-600 mt-2">
          Red arrows: positive gradients, Blue arrows: negative gradients
        </p>
      </div>
    </div>
  );
}

