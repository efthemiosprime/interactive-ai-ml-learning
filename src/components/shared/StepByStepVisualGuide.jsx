import React, { useState } from 'react';
import { ChevronRight, ArrowDown, ArrowRight } from 'lucide-react';

export default function StepByStepVisualGuide({ steps, title, color = 'indigo' }) {
  const [currentStep, setCurrentStep] = useState(0);

  const colorClasses = {
    indigo: {
      bg: 'bg-indigo-50',
      border: 'border-indigo-200',
      text: 'text-indigo-900',
      button: 'bg-indigo-600 hover:bg-indigo-700',
      stepBg: 'bg-indigo-100',
      stepActive: 'bg-indigo-500'
    },
    purple: {
      bg: 'bg-purple-50',
      border: 'border-purple-200',
      text: 'text-purple-900',
      button: 'bg-purple-600 hover:bg-purple-700',
      stepBg: 'bg-purple-100',
      stepActive: 'bg-purple-500'
    },
    blue: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-900',
      button: 'bg-blue-600 hover:bg-blue-700',
      stepBg: 'bg-blue-100',
      stepActive: 'bg-blue-500'
    }
  };

  const colors = colorClasses[color] || colorClasses.indigo;

  return (
    <div className={`${colors.bg} rounded-xl p-6 border-2 ${colors.border}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`text-xl font-bold ${colors.text}`}>{title}</h3>
        <div className="flex gap-2">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className={`px-3 py-1 rounded ${colors.button} text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            ← Prev
          </button>
          <button
            onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
            className={`px-3 py-1 rounded ${colors.button} text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            Next →
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {/* Step Indicator */}
        <div className="flex items-center justify-center gap-2 mb-4">
          {steps.map((_, idx) => (
            <React.Fragment key={idx}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all ${
                  idx === currentStep
                    ? `${colors.stepActive} text-white scale-110 ring-2 ring-white`
                    : idx < currentStep
                    ? `${colors.stepActive} text-white opacity-60`
                    : `${colors.stepBg} ${colors.text}`
                }`}
              >
                {idx + 1}
              </div>
              {idx < steps.length - 1 && (
                <ChevronRight className={`w-5 h-5 ${idx < currentStep ? colors.text : 'text-gray-300'}`} />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Current Step Content */}
        <div className={`bg-white rounded-lg p-6 border-2 ${colors.border} min-h-[200px]`}>
          <div className="flex items-start gap-4">
            <div className={`flex-shrink-0 w-12 h-12 ${colors.stepActive} text-white rounded-lg flex items-center justify-center font-bold text-lg`}>
              {currentStep + 1}
            </div>
            <div className="flex-1">
              <h4 className={`text-lg font-bold ${colors.text} mb-2`}>
                {steps[currentStep].title}
              </h4>
              <p className="text-gray-700 mb-4">
                {steps[currentStep].description}
              </p>
              {steps[currentStep].formula && (
                <div className={`${colors.bg} rounded-lg p-3 mb-3`}>
                  <div className="font-mono text-sm font-semibold">
                    {steps[currentStep].formula}
                  </div>
                </div>
              )}
              {steps[currentStep].visual && (
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  {steps[currentStep].visual}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`${colors.stepActive} h-2 rounded-full transition-all duration-300`}
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}

