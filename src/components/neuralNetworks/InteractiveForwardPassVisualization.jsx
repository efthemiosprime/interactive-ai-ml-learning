import React, { useRef, useEffect, useState } from 'react';
import { Play } from 'lucide-react';
import * as nn from '../../utils/neuralNetworks';

export default function InteractiveForwardPassVisualization() {
  const canvasRef = useRef(null);
  const [input, setInput] = useState([0.5, -0.3, 0.8]);
  const [network, setNetwork] = useState(() => 
    nn.createNetwork([3, 4, 2], ['relu', 'sigmoid'])
  );
  const [result, setResult] = useState(null);
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

    // Calculate forward pass
    const forwardResult = nn.forwardPassNetwork(input, network);
    const activations = forwardResult.activations;

    // Draw layers
    const neuronPositions = [];
    activations.forEach((layerActivations, layerIdx) => {
      const x = padding + layerSpacing * layerIdx;
      const numNeurons = layerActivations.length;
      const neuronSpacing = (height - 2 * padding) / (numNeurons + 1);
      const layerNeurons = [];

      for (let i = 0; i < numNeurons; i++) {
        const y = padding + neuronSpacing * (i + 1);
        layerNeurons.push({ x, y, value: layerActivations[i] });
      }

      neuronPositions.push(layerNeurons);
    });

    // Draw connections with weights
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let l = 0; l < neuronPositions.length - 1; l++) {
      const currentLayer = neuronPositions[l];
      const nextLayer = neuronPositions[l + 1];
      const layer = network[l];

      currentLayer.forEach((neuron1, i) => {
        nextLayer.forEach((neuron2, j) => {
          const weight = layer.weights[j][i];
          const opacity = Math.abs(weight) * 2;
          ctx.strokeStyle = `rgba(139, 92, 246, ${Math.min(opacity, 1)})`;
          ctx.lineWidth = Math.abs(weight) * 3 + 0.5;
          ctx.beginPath();
          ctx.moveTo(neuron1.x, neuron1.y);
          ctx.lineTo(neuron2.x, neuron2.y);
          ctx.stroke();
        });
      });
    }

    // Draw neurons with activation values
    neuronPositions.forEach((layer, layerIdx) => {
      layer.forEach((neuron) => {
        // Color based on activation value
        const intensity = Math.max(0, Math.min(1, Math.abs(neuron.value)));
        const hue = neuron.value > 0 ? 240 : 0;
        ctx.fillStyle = `hsla(${hue}, 70%, ${50 + intensity * 30}%, ${0.3 + intensity * 0.7})`;

        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, 15, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw value
        ctx.fillStyle = '#1f2937';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(neuron.value.toFixed(2), neuron.x, neuron.y + 3);
      });
    });

    // Draw labels
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Forward Pass: Data Flow Through Network', width / 2, 25);

    // Draw layer labels
    const layerLabels = ['Input', ...Array(network.length - 1).fill('Hidden'), 'Output'];
    neuronPositions.forEach((layer, idx) => {
      ctx.fillText(layerLabels[idx], layer[0].x, layer[0].y - 30);
    });

    setResult(forwardResult);
  }, [input, network]);

  const runForwardPass = () => {
    setIsAnimating(true);
    setTimeout(() => {
      setIsAnimating(false);
    }, 100);
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> See how input values propagate through the network layers!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Input Values:
          </label>
          <div className="grid grid-cols-3 gap-2">
            {input.map((val, idx) => (
              <div key={idx}>
                <label className="text-xs text-gray-600">Input {idx + 1}:</label>
                <input
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
                <span className="text-xs text-gray-600">{val.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={runForwardPass}
          disabled={isAnimating}
          className="w-full px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Play className="w-4 h-4" />
          Run Forward Pass
        </button>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Output */}
      {result && (
        <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
          <h4 className="font-semibold text-green-900 mb-2">Output:</h4>
          <div className="flex gap-4">
            {result.output.map((val, idx) => (
              <div key={idx} className="bg-white rounded p-2">
                <div className="text-xs text-gray-600">Output {idx + 1}</div>
                <div className="text-lg font-bold text-green-600">
                  {val.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Forward Pass Steps:</h4>
        <ol className="text-sm text-gray-700 space-y-1 list-decimal list-inside">
          <li>Input values enter the input layer</li>
          <li>Each neuron computes: weighted sum + bias</li>
          <li>Activation function is applied</li>
          <li>Result propagates to next layer</li>
          <li>Process repeats until output layer</li>
        </ol>
      </div>
    </div>
  );
}

