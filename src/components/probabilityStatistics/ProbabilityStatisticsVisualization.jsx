import React, { useMemo } from 'react';
import * as stats from '../../utils/probabilityStatistics';
import InteractiveDescriptiveStatsVisualization from './InteractiveDescriptiveStatsVisualization';
import InteractiveCovarianceVisualization from './InteractiveCovarianceVisualization';
import InteractiveBayesVisualization from './InteractiveBayesVisualization';
import InteractiveDistributionsVisualization from './InteractiveDistributionsVisualization';

export default function ProbabilityStatisticsVisualization({ selectedTopic, dataSet }) {
  const descriptiveStats = useMemo(() => {
    if (dataSet.length > 0) {
      return stats.calculateDescriptiveStats(dataSet);
    }
    return null;
  }, [dataSet]);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Visualization</h2>

      {selectedTopic === 'descriptive' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Descriptive Statistics</h3>
          {descriptiveStats && (
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Mean</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.mean.toFixed(3)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Median</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.median.toFixed(3)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Variance</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.variance.toFixed(3)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Standard Deviation</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.standardDeviation.toFixed(3)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Min</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.min.toFixed(3)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Max</div>
                <div className="text-2xl font-bold text-blue-600">
                  {descriptiveStats.max.toFixed(3)}
                </div>
              </div>
            </div>
          )}
          <InteractiveDescriptiveStatsVisualization dataSet={dataSet} />
        </div>
      )}

      {selectedTopic === 'covariance' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Covariance & Correlation</h3>
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-gray-700 mb-4">
              Covariance measures how two variables change together. Correlation normalizes 
              this to a range of -1 to 1.
            </p>
            <div className="bg-white p-4 rounded border-2 border-blue-200">
              <div className="font-mono text-sm space-y-2">
                <div>Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]</div>
                <div>Corr(X, Y) = Cov(X, Y) / (σₓ × σᵧ)</div>
              </div>
            </div>
            <p className="text-sm text-gray-600 mt-4">
              In ML, covariance matrices are used in Principal Component Analysis (PCA) 
              to find directions of maximum variance.
            </p>
          </div>
          <InteractiveCovarianceVisualization />
        </div>
      )}

      {selectedTopic === 'conditional-probability' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Conditional Probability</h3>
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-gray-700 mb-4">
              Conditional probability P(A|B) is the probability of A given that B has occurred.
            </p>
            <div className="bg-white p-4 rounded border-2 border-blue-200">
              <div className="font-mono text-lg mb-2">
                P(A|B) = P(A ∩ B) / P(B)
              </div>
            </div>
            <p className="text-sm text-gray-600 mt-4">
              Used in probabilistic models, decision trees, and understanding feature dependencies.
            </p>
          </div>
        </div>
      )}

      {selectedTopic === 'bayes' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Bayes' Theorem</h3>
          <InteractiveBayesVisualization />
        </div>
      )}

      {selectedTopic === 'distributions' && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Probability Distributions</h3>
          <InteractiveDistributionsVisualization />
        </div>
      )}
    </div>
  );
}

