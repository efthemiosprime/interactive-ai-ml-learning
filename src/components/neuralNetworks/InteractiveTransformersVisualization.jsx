import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import * as nn from '../../utils/neuralNetworks';

export default function InteractiveTransformersVisualization() {
  const canvasRef = useRef(null);
  const [numTokens, setNumTokens] = useState(4);
  const [attentionResult, setAttentionResult] = useState(null);

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

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Generate random query, keys, values
    const query = Array(3).fill(0).map(() => Math.random() - 0.5);
    const keys = Array(numTokens).fill(0).map(() => 
      Array(3).fill(0).map(() => Math.random() - 0.5)
    );
    const values = Array(numTokens).fill(0).map(() => 
      Array(3).fill(0).map(() => Math.random() - 0.5)
    );

    // Calculate attention
    const result = nn.calculateAttention(query, keys, values);
    setAttentionResult(result);

    // Draw tokens (keys/values)
    const tokenSpacing = (width - 2 * padding) / (numTokens + 1);
    const tokenY = height / 2;
    const tokenRadius = 20;

    // Draw query
    const queryX = padding;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(queryX, tokenY, tokenRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Q', queryX, tokenY + 4);

    // Draw keys and attention weights
    keys.forEach((key, idx) => {
      const x = padding + tokenSpacing * (idx + 1);
      
      // Draw attention weight as line thickness
      const weight = result.attentionWeights[idx];
      const lineWidth = weight * 10 + 1;
      const opacity = weight;

      // Draw connection from query to key
      ctx.strokeStyle = `rgba(139, 92, 246, ${opacity})`;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(queryX + tokenRadius, tokenY);
      ctx.lineTo(x - tokenRadius, tokenY);
      ctx.stroke();

      // Draw key token
      ctx.fillStyle = `rgba(239, 68, 68, ${0.3 + opacity * 0.7})`;
      ctx.beginPath();
      ctx.arc(x, tokenY - 60, tokenRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`K${idx + 1}`, x, tokenY - 60 + 4);

      // Draw value token
      ctx.fillStyle = `rgba(16, 185, 129, ${0.3 + opacity * 0.7})`;
      ctx.beginPath();
      ctx.arc(x, tokenY + 60, tokenRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`V${idx + 1}`, x, tokenY + 60 + 4);

      // Draw attention weight
      ctx.fillStyle = '#1f2937';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(weight.toFixed(2), x, tokenY - 90);
    });

    // Draw output
    const outputX = width - padding;
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(outputX, tokenY, tokenRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText('Out', outputX, tokenY + 4);

    // Draw connections from values to output
    values.forEach((val, idx) => {
      const x = padding + tokenSpacing * (idx + 1);
      const weight = result.attentionWeights[idx];
      ctx.strokeStyle = `rgba(16, 185, 129, ${weight})`;
      ctx.lineWidth = weight * 10 + 1;
      ctx.beginPath();
      ctx.moveTo(x, tokenY + 60 + tokenRadius);
      ctx.lineTo(outputX - tokenRadius, tokenY);
      ctx.stroke();
    });

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Self-Attention Mechanism (Transformer)', width / 2, 25);

    // Draw labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Query', queryX, tokenY - 30);
    ctx.fillText('Keys', width / 2, tokenY - 120);
    ctx.fillText('Values', width / 2, tokenY + 120);
    ctx.fillText('Output', outputX, tokenY - 30);
  }, [numTokens]);

  return (
    <div className="space-y-4">
      <div className="bg-violet-50 border-2 border-violet-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-violet-800">
          💡 <strong>Interactive:</strong> See how attention weights determine which tokens to focus on!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-violet-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Number of Tokens: {numTokens}
        </label>
        <input
          type="range"
          min="2"
          max="6"
          step="1"
          value={numTokens}
          onChange={(e) => setNumTokens(parseInt(e.target.value))}
          className="w-full"
        />
        <button
          onClick={() => setNumTokens(numTokens)}
          className="mt-2 w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Regenerate Attention
        </button>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Attention weights */}
      {attentionResult && (
        <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
          <h4 className="font-semibold text-green-900 mb-2">Attention Weights:</h4>
          <div className="flex gap-2 flex-wrap">
            {attentionResult.attentionWeights.map((weight, idx) => (
              <div key={idx} className="bg-white rounded p-2">
                <div className="text-xs text-gray-600">Token {idx + 1}</div>
                <div className="text-lg font-bold text-green-600">
                  {weight.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Attention Mechanism:</h4>
        <ol className="text-sm text-gray-700 space-y-1 list-decimal list-inside">
          <li><strong>Query (Q):</strong> What we're looking for</li>
          <li><strong>Key (K):</strong> What each token offers</li>
          <li><strong>Value (V):</strong> The actual content of each token</li>
          <li><strong>Attention Score:</strong> Dot product of Q and K (how relevant)</li>
          <li><strong>Softmax:</strong> Converts scores to probabilities (attention weights)</li>
          <li><strong>Output:</strong> Weighted sum of values based on attention weights</li>
        </ol>
        <p className="text-xs text-gray-600 mt-2">
          This is the core mechanism behind transformers and modern LLMs like GPT!
        </p>
      </div>
    </div>
  );
}

