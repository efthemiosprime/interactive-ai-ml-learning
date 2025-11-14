import React, { useRef, useEffect, useState } from 'react';
import { Play, RefreshCw } from 'lucide-react';
import * as nn from '../../utils/neuralNetworks';

export default function InteractiveTrainingVisualization() {
  const canvasRef = useRef(null);
  const [network, setNetwork] = useState(() => 
    nn.createNetwork([1, 4, 1], ['relu', 'linear'])
  );
  const [trainingData, setTrainingData] = useState(() => 
    nn.generateTrainingData(10)
  );
  const [lossHistory, setLossHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

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
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw data points
    const xMin = -1;
    const xMax = 1;
    const yMin = -1.5;
    const yMax = 1.5;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
    const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = padding.left + (i / 4) * (width - padding.left - padding.right);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    const zeroY = toScreenY(0);
    const zeroX = toScreenX(0);
    ctx.beginPath();
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(width - padding.right, zeroY);
    ctx.moveTo(zeroX, padding.top);
    ctx.lineTo(zeroX, height - padding.bottom);
    ctx.stroke();

    // Draw training data points
    trainingData.forEach(({ input, target }) => {
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.arc(toScreenX(input[0]), toScreenY(target[0]), 5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw model predictions
    if (lossHistory.length > 0) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const step = 0.02;
      let firstPoint = true;
      for (let x = xMin; x <= xMax; x += step) {
        const result = nn.forwardPassNetwork([x], network);
        const y = result.output[0];
        const screenX = toScreenX(x);
        const screenY = toScreenY(y);

        if (firstPoint) {
          ctx.moveTo(screenX, screenY);
          firstPoint = false;
        } else {
          ctx.lineTo(screenX, screenY);
        }
      }
      ctx.stroke();
    }

    // Draw loss history (if available)
    if (lossHistory.length > 1) {
      const lossMax = Math.max(...lossHistory);
      const lossMin = Math.min(...lossHistory);
      const lossRange = lossMax - lossMin || 1;

      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      lossHistory.forEach((loss, idx) => {
        const x = padding.left + (idx / (lossHistory.length - 1)) * (width - padding.left - padding.right);
        const y = padding.top + 20 + ((loss - lossMin) / lossRange) * 50;
        
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Training: Learning from Data', width / 2, 20);

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Input (x)', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Output (y)', 0, 0);
    ctx.restore();
  }, [network, trainingData, lossHistory]);

  const trainStep = () => {
    setIsTraining(true);
    
    let currentNetwork = JSON.parse(JSON.stringify(network));
    const newLossHistory = [...lossHistory];
    const learningRate = 0.1;
    
    // Train for a few steps
    for (let step = 0; step < 5; step++) {
      let totalLoss = 0;
      
      trainingData.forEach(({ input, target }) => {
        const forwardResult = nn.forwardPassNetwork(input, currentNetwork);
        const loss = nn.mseLoss(forwardResult.output, target);
        totalLoss += loss;
        
        const gradients = nn.backpropagate(
          currentNetwork,
          forwardResult.activations,
          target,
          'mse'
        );
        
        nn.updateWeights(currentNetwork, gradients, learningRate);
      });
      
      newLossHistory.push(totalLoss / trainingData.length);
    }
    
    setNetwork(currentNetwork);
    setLossHistory(newLossHistory);
    setEpoch(epoch + 5);
    setIsTraining(false);
  };

  const resetTraining = () => {
    setNetwork(nn.createNetwork([1, 4, 1], ['relu', 'linear']));
    setLossHistory([]);
    setEpoch(0);
  };

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> Watch the model learn to fit the data as training progresses!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded p-3">
            <div className="text-sm text-gray-600">Epoch</div>
            <div className="text-2xl font-bold text-blue-600">{epoch}</div>
          </div>
          <div className="bg-red-50 rounded p-3">
            <div className="text-sm text-gray-600">Loss</div>
            <div className="text-2xl font-bold text-red-600">
              {lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].toFixed(4) : '0.0000'}
            </div>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={trainStep}
            disabled={isTraining}
            className="flex-1 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4" />
            Train (5 epochs)
          </button>
          <button
            onClick={resetTraining}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Training Process:</h4>
        <ol className="text-sm text-gray-700 space-y-1 list-decimal list-inside">
          <li>Initialize network with random weights</li>
          <li>Forward pass: Make predictions on training data</li>
          <li>Calculate loss: Compare predictions to targets</li>
          <li>Backpropagation: Compute gradients</li>
          <li>Update weights: Move in direction that reduces loss</li>
          <li>Repeat until loss converges</li>
        </ol>
        <p className="text-xs text-gray-600 mt-2">
          Blue dots: Training data, Green line: Model predictions, Red line: Loss over time
        </p>
      </div>
    </div>
  );
}

