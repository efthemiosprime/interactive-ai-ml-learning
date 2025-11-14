import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function UnsupervisedLearningEducationalPanels({ selectedTopic }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'clustering' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-teal-900 mb-3">Clustering Algorithms</h3>
            <p className="text-gray-700 mb-4">
              Clustering groups similar data points together without labeled examples. 
              It discovers hidden patterns and structures in data.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">K-Means Clustering</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Partitions data into k clusters</li>
              <li>Minimizes within-cluster variance</li>
              <li>Iteratively updates centroids until convergence</li>
              <li>Requires specifying number of clusters beforehand</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Hierarchical Clustering</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Builds a tree of clusters (dendrogram)</li>
              <li>Agglomerative: Start with individual points, merge clusters</li>
              <li>Divisive: Start with all points, split clusters</li>
              <li>No need to specify number of clusters</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">DBSCAN</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Density-based clustering algorithm</li>
              <li>Groups closely packed points together</li>
              <li>Marks outliers as noise</li>
              <li>Can find clusters of arbitrary shape</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="unsupervised-learning" operationType="clustering" />
        </div>
      )}

      {selectedTopic === 'dimensionality-reduction' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-teal-900 mb-3">Dimensionality Reduction</h3>
            <p className="text-gray-700 mb-4">
              Dimensionality reduction reduces the number of features while preserving important information. 
              Essential for visualization, noise reduction, and computational efficiency.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Principal Component Analysis (PCA)</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Finds directions of maximum variance (principal components)</li>
              <li>Uses eigenvalues and eigenvectors of covariance matrix</li>
              <li>Projects data onto lower-dimensional space</li>
              <li>Linear transformation, preserves global structure</li>
            </ul>
            <div className="bg-teal-50 rounded-lg p-4 mb-4 border-2 border-teal-200">
              <div className="font-mono text-sm">
                <div><strong>Mathematical Form:</strong> Y = XW</div>
                <div className="text-xs text-gray-600 mt-2">
                  Where W contains eigenvectors (principal components) as columns
                </div>
              </div>
            </div>

            <h4 className="font-semibold text-gray-800 mb-2">t-SNE (t-Distributed Stochastic Neighbor Embedding)</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Non-linear dimensionality reduction</li>
              <li>Preserves local neighborhood structure</li>
              <li>Excellent for visualization (2D/3D)</li>
              <li>Stochastic algorithm, results may vary</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Autoencoders</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Neural network that learns compressed representation</li>
              <li>Encoder compresses, decoder reconstructs</li>
              <li>Non-linear transformation</li>
              <li>Can learn complex patterns</li>
            </ul>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="unsupervised-learning" operationType="dimensionality-reduction" />
        </div>
      )}

      {selectedTopic === 'anomaly-detection' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-teal-900 mb-3">Anomaly Detection</h3>
            <p className="text-gray-700 mb-4">
              Anomaly detection identifies unusual patterns, outliers, or rare events in data. 
              Critical for fraud detection, quality control, and security.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Isolation Forest</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Randomly selects features and split values</li>
              <li>Anomalies are easier to isolate (fewer splits needed)</li>
              <li>Efficient for high-dimensional data</li>
              <li>Doesn't require normal data distribution assumptions</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">One-Class SVM</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Learns a boundary around normal data</li>
              <li>Points outside the boundary are anomalies</li>
              <li>Uses kernel trick for non-linear boundaries</li>
              <li>Requires mostly normal training data</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Mathematical Connection</h4>
            <p className="text-gray-700 mb-4">
              Anomaly detection uses distance metrics and density estimation. 
              Points far from the data distribution or in low-density regions are flagged as anomalies.
            </p>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="unsupervised-learning" operationType="anomaly-detection" />
        </div>
      )}

      {selectedTopic === 'distance-metrics' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-teal-900 mb-3">Distance Metrics</h3>
            <p className="text-gray-700 mb-4">
              Distance metrics measure similarity or dissimilarity between data points. 
              Essential for clustering, nearest neighbor algorithms, and similarity search.
            </p>
            
            <h4 className="font-semibold text-gray-800 mb-2">Euclidean Distance (L2)</h4>
            <div className="bg-teal-50 rounded-lg p-4 mb-4 border-2 border-teal-200">
              <div className="font-mono text-sm">
                <div><strong>Formula:</strong> d = √(Σ(xᵢ - yᵢ)²)</div>
                <div className="text-xs text-gray-600 mt-2">
                  Straight-line distance between two points in Euclidean space
                </div>
              </div>
            </div>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Most common distance metric</li>
              <li>Sensitive to outliers</li>
              <li>Used in K-means, KNN</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Manhattan Distance (L1)</h4>
            <div className="bg-teal-50 rounded-lg p-4 mb-4 border-2 border-teal-200">
              <div className="font-mono text-sm">
                <div><strong>Formula:</strong> d = Σ|xᵢ - yᵢ|</div>
                <div className="text-xs text-gray-600 mt-2">
                  Sum of absolute differences (city block distance)
                </div>
              </div>
            </div>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Robust to outliers</li>
              <li>Used when features are independent</li>
              <li>Common in grid-based paths</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Cosine Distance</h4>
            <div className="bg-teal-50 rounded-lg p-4 mb-4 border-2 border-teal-200">
              <div className="font-mono text-sm">
                <div><strong>Formula:</strong> d = 1 - cos(θ) = 1 - (A·B) / (||A|| × ||B||)</div>
                <div className="text-xs text-gray-600 mt-2">
                  Measures angle between vectors, ignores magnitude
                </div>
              </div>
            </div>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li>Used for text similarity, document comparison</li>
              <li>Normalized (range 0 to 1)</li>
              <li>Ignores vector magnitude, focuses on direction</li>
            </ul>

            <h4 className="font-semibold text-gray-800 mb-2">Connection to Linear Algebra</h4>
            <p className="text-gray-700 mb-4">
              Distance metrics are fundamental operations in vector spaces. 
              They use dot products, norms, and vector operations from linear algebra.
            </p>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="unsupervised-learning" operationType="distance-metrics" />
        </div>
      )}
    </div>
  );
}

