import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function ProbabilityStatisticsEducationalPanels({ selectedTopic, dataSet }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'descriptive' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Descriptive Statistics</h3>
            <p className="text-gray-700 mb-4">
              Descriptive statistics summarize and describe the main features of a dataset. 
              They're essential for understanding data before applying ML algorithms.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Measures:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Mean:</strong> Average value, sensitive to outliers</li>
              <li><strong>Median:</strong> Middle value, robust to outliers</li>
              <li><strong>Variance:</strong> Measures spread around the mean</li>
              <li><strong>Standard Deviation:</strong> Square root of variance, same units as data</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="descriptive" />
        </div>
      )}

      {selectedTopic === 'covariance' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Covariance & Correlation</h3>
            <p className="text-gray-700 mb-4">
              Covariance measures how two variables change together. Correlation is the normalized 
              version, ranging from -1 (perfect negative) to +1 (perfect positive).
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Positive Covariance:</strong> Variables increase together</li>
              <li><strong>Negative Covariance:</strong> One increases as the other decreases</li>
              <li><strong>Zero Covariance:</strong> Variables are independent</li>
              <li><strong>Correlation:</strong> Normalized covariance (-1 to 1)</li>
            </ul>
            <h4 className="font-semibold text-gray-800 mt-4 mb-2">ML Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>PCA:</strong> Uses covariance matrix to find principal components</li>
              <li><strong>Feature Selection:</strong> Remove highly correlated features</li>
              <li><strong>Multivariate Analysis:</strong> Understanding feature relationships</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'conditional-probability' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Conditional Probability</h3>
            <p className="text-gray-700 mb-4">
              Conditional probability P(A|B) represents the probability of event A occurring 
              given that event B has occurred. It's fundamental to probabilistic models.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Dependence:</strong> Events are dependent if P(A|B) ≠ P(A)</li>
              <li><strong>Independence:</strong> Events are independent if P(A|B) = P(A)</li>
              <li><strong>Chain Rule:</strong> P(A∩B) = P(A|B) × P(B)</li>
            </ul>
            <h4 className="font-semibold text-gray-800 mt-4 mb-2">ML Applications:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Probabilistic Models:</strong> Modeling dependencies between variables</li>
              <li><strong>Decision Trees:</strong> Splitting based on conditional probabilities</li>
              <li><strong>Bayesian Networks:</strong> Representing conditional dependencies</li>
            </ul>
          </div>
        </div>
      )}

      {selectedTopic === 'bayes' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Bayes' Theorem</h3>
            <p className="text-gray-700 mb-4">
              Bayes' theorem provides a way to update probabilities based on new evidence. 
              It's the foundation of Bayesian inference and many ML algorithms.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Components:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Prior:</strong> Initial belief about probability</li>
              <li><strong>Likelihood:</strong> Probability of evidence given hypothesis</li>
              <li><strong>Posterior:</strong> Updated probability after seeing evidence</li>
              <li><strong>Evidence:</strong> Normalizing constant</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="bayes" />
        </div>
      )}

      {selectedTopic === 'distributions' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Probability Distributions</h3>
            <p className="text-gray-700 mb-4">
              Probability distributions describe how probabilities are distributed over possible outcomes. 
              Different distributions model different types of data and uncertainty.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Common Distributions:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Normal:</strong> Continuous, symmetric, many natural phenomena</li>
              <li><strong>Bernoulli:</strong> Binary outcomes (0 or 1)</li>
              <li><strong>Binomial:</strong> Number of successes in n trials</li>
              <li><strong>Poisson:</strong> Count of events in fixed interval</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="distributions" />
        </div>
      )}
    </div>
  );
}

