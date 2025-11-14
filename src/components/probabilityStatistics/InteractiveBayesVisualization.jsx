import React, { useState } from 'react';
import * as stats from '../../utils/probabilityStatistics';

export default function InteractiveBayesVisualization() {
  const [prior, setPrior] = useState(0.3);
  const [likelihood, setLikelihood] = useState(0.8);
  const [evidence, setEvidence] = useState(0.32);

  const posterior = stats.applyBayesTheorem(likelihood, prior, evidence);

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Adjust the probabilities below to see how Bayes' theorem updates beliefs!
        </p>
      </div>

      {/* Interactive Controls */}
      <div className="bg-white rounded-lg p-6 border-2 border-blue-200">
        <h4 className="font-bold text-blue-900 mb-4 text-center">Interactive Bayes' Calculator</h4>
        
        <div className="space-y-4">
          {/* Prior P(A) */}
          <div>
            <label className="block text-sm font-semibold text-blue-900 mb-2">
              Prior P(A): {prior.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={prior}
              onChange={(e) => setPrior(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
            <div className="mt-2 h-3 bg-blue-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-600 transition-all duration-300"
                style={{ width: `${prior * 100}%` }}
              />
            </div>
          </div>

          {/* Likelihood P(B|A) */}
          <div>
            <label className="block text-sm font-semibold text-purple-900 mb-2">
              Likelihood P(B|A): {likelihood.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={likelihood}
              onChange={(e) => setLikelihood(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
            <div className="mt-2 h-3 bg-purple-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-purple-600 transition-all duration-300"
                style={{ width: `${likelihood * 100}%` }}
              />
            </div>
          </div>

          {/* Evidence P(B) */}
          <div>
            <label className="block text-sm font-semibold text-yellow-900 mb-2">
              Evidence P(B): {evidence.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={evidence}
              onChange={(e) => setEvidence(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>0</span>
              <span>0.5</span>
              <span>1.0</span>
            </div>
            <div className="mt-2 h-3 bg-yellow-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-yellow-600 transition-all duration-300"
                style={{ width: `${evidence * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Formula Display */}
        <div className="mt-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border-2 border-green-300">
          <div className="text-center mb-3">
            <div className="font-mono text-xl font-bold text-green-900 mb-2">
              P(A|B) = P(B|A) × P(A) / P(B)
            </div>
            <div className="font-mono text-lg text-green-800 mt-2">
              = {likelihood.toFixed(2)} × {prior.toFixed(2)} / {evidence.toFixed(2)}
            </div>
            <div className="font-mono text-2xl font-bold text-green-900 mt-3">
              = {posterior.toFixed(3)}
            </div>
          </div>

          {/* Posterior Visualization */}
          <div className="mt-4">
            <label className="block text-sm font-semibold text-green-900 mb-2">
              Posterior P(A|B): {posterior.toFixed(3)}
            </label>
            <div className="h-6 bg-green-200 rounded-full overflow-hidden relative">
              <div 
                className="h-full bg-green-600 transition-all duration-300 flex items-center justify-center"
                style={{ width: `${Math.min(posterior * 100, 100)}%` }}
              >
                {posterior > 0.1 && (
                  <span className="text-white text-xs font-bold">
                    {posterior.toFixed(3)}
                  </span>
                )}
              </div>
              {posterior <= 0.1 && (
                <span className="absolute left-2 top-1/2 transform -translate-y-1/2 text-green-800 text-xs font-bold">
                  {posterior.toFixed(3)}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Step-by-Step Calculation */}
        <div className="mt-6 bg-white rounded-lg p-4 border border-blue-200">
          <h5 className="font-semibold text-blue-900 mb-3">Step-by-Step Calculation:</h5>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">1</span>
              <span>Start with prior: <strong>P(A) = {prior.toFixed(2)}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">2</span>
              <span>Calculate likelihood: <strong>P(B|A) = {likelihood.toFixed(2)}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-yellow-500 text-white rounded-full flex items-center justify-center text-xs font-bold">3</span>
              <span>Calculate evidence: <strong>P(B) = {evidence.toFixed(2)}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold">4</span>
              <span className="font-mono">
                Posterior = ({likelihood.toFixed(2)} × {prior.toFixed(2)}) / {evidence.toFixed(2)} = <strong>{posterior.toFixed(3)}</strong>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

