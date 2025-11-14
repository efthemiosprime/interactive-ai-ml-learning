import React, { useMemo, useState, useEffect } from 'react';
import * as math from '../../utils/math';
import StepByStepVisualGuide from '../shared/StepByStepVisualGuide';
import { ArrowDown, ArrowRight, TrendingDown } from 'lucide-react';
import InteractiveDerivativeVisualization from './InteractiveDerivativeVisualization';
import InteractivePartialDerivativesVisualization from './InteractivePartialDerivativesVisualization';
import InteractiveGradientVisualization from './InteractiveGradientVisualization';
import InteractiveChainRuleVisualization from './InteractiveChainRuleVisualization';
import InteractiveBackpropagationVisualization from './InteractiveBackpropagationVisualization';

export default function CalculusVisualization({ selectedTopic, functionType }) {
  const [showStepByStep, setShowStepByStep] = useState(false);
  const getFunction = (type) => {
    switch (type) {
      case 'quadratic':
        return (x) => x * x;
      case 'cubic':
        return (x) => x * x * x;
      case 'sine':
        return (x) => Math.sin(x);
      case 'exponential':
        return (x) => Math.exp(x);
      default:
        return (x) => x * x;
    }
  };

  const getDerivative = (type) => {
    switch (type) {
      case 'quadratic':
        return (x) => 2 * x;
      case 'cubic':
        return (x) => 3 * x * x;
      case 'sine':
        return (x) => Math.cos(x);
      case 'exponential':
        return (x) => Math.exp(x);
      default:
        return (x) => 2 * x;
    }
  };

  const f = useMemo(() => getFunction(functionType), [functionType]);
  const fPrime = useMemo(() => getDerivative(functionType), [functionType]);

  const samplePoints = useMemo(() => {
    const points = [];
    for (let x = -5; x <= 5; x += 0.5) {
      points.push({ x, y: f(x), derivative: fPrime(x) });
    }
    return points;
  }, [f, fPrime]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'derivatives' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Function: f(x) = {
                functionType === 'quadratic' && 'x²'
              }
              {functionType === 'cubic' && 'x³'}
              {functionType === 'sine' && 'sin(x)'}
              {functionType === 'exponential' && 'eˣ'}
            </h3>
            <InteractiveDerivativeVisualization functionType={functionType} />
          </div>
        </div>
      )}

      {selectedTopic === 'partial-derivatives' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Partial Derivatives</h3>
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-gray-700 mb-4">
              For a function f(x, y), partial derivatives measure how the function changes 
              with respect to one variable while keeping others constant.
            </p>
            <div className="bg-white p-4 rounded border-2 border-purple-200">
              <div className="font-mono text-sm space-y-2">
                <div>∂f/∂x = lim(h→0) [f(x+h, y) - f(x, y)] / h</div>
                <div>∂f/∂y = lim(h→0) [f(x, y+h) - f(x, y)] / h</div>
              </div>
            </div>
          </div>
          <InteractivePartialDerivativesVisualization />
        </div>
      )}

      {selectedTopic === 'gradients' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Gradients & Gradient Descent</h3>
          <InteractiveGradientVisualization />
        </div>
      )}

      {selectedTopic === 'chain-rule' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Chain Rule</h3>
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-gray-700 mb-4">
              The chain rule allows us to compute derivatives of composite functions. 
              This is essential for backpropagation in neural networks.
            </p>
            <div className="bg-white p-4 rounded border-2 border-purple-200">
              <div className="font-mono text-sm mb-2">
                d/dx [f(g(x))] = f'(g(x)) × g'(x)
              </div>
              <p className="text-sm text-gray-600 mt-2">
                For multiple layers: ∂L/∂w = (∂L/∂y) × (∂y/∂z) × (∂z/∂w)
              </p>
            </div>
          </div>
          <InteractiveChainRuleVisualization />
        </div>
      )}

      {selectedTopic === 'backpropagation' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Backpropagation</h3>
          <InteractiveBackpropagationVisualization />
        </div>
      )}
    </div>
  );
}

