import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function PCAVisualization() {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [eigenvectors, setEigenvectors] = useState([]);
  const [eigenvalues, setEigenvalues] = useState([]);
  const [principalComponents, setPrincipalComponents] = useState([]);
  const [showPCA, setShowPCA] = useState(false);
  const [dimension, setDimension] = useState(2);

  useEffect(() => {
    generateSampleData();
  }, []);

  useEffect(() => {
    if (dataPoints.length > 0) {
      calculatePCA();
    }
  }, [dataPoints]);

  useEffect(() => {
    drawGraph();
  }, [dataPoints, eigenvectors, eigenvalues, principalComponents, showPCA, dimension]);

  const generateSampleData = () => {
    const points = [];
    const centerX = 0;
    const centerY = 0;
    
    // Generate correlated 2D data
    for (let i = 0; i < 50; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 3;
      const x = centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 2;
      const y = centerY + radius * Math.sin(angle) * 0.5 + (Math.random() - 0.5) * 2;
      points.push({ x, y });
    }
    
    setDataPoints(points);
    setShowPCA(false);
  };

  const calculatePCA = () => {
    if (dataPoints.length === 0) return;

    // Step 1: Center the data (subtract mean)
    const meanX = dataPoints.reduce((sum, p) => sum + p.x, 0) / dataPoints.length;
    const meanY = dataPoints.reduce((sum, p) => sum + p.y, 0) / dataPoints.length;
    
    const centered = dataPoints.map(p => ({
      x: p.x - meanX,
      y: p.y - meanY
    }));

    // Step 2: Calculate covariance matrix
    // Cov(X,Y) = (1/n) Σ(x_i - x̄)(y_i - ȳ)
    const n = centered.length;
    const covXX = centered.reduce((sum, p) => sum + p.x * p.x, 0) / n;
    const covYY = centered.reduce((sum, p) => sum + p.y * p.y, 0) / n;
    const covXY = centered.reduce((sum, p) => sum + p.x * p.y, 0) / n;

    const covarianceMatrix = [
      [covXX, covXY],
      [covXY, covYY]
    ];

    // Step 3: Calculate eigenvalues and eigenvectors
    // For 2x2 matrix: eigenvalues are solutions to det(C - λI) = 0
    const trace = covXX + covYY;
    const det = covXX * covYY - covXY * covXY;
    
    const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
    const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;

    // Eigenvectors: solve (C - λI)v = 0
    let eigenvec1, eigenvec2;
    
    if (Math.abs(covXY) > 0.001) {
      // First eigenvector
      const v1x = lambda1 - covYY;
      const v1y = covXY;
      const len1 = Math.sqrt(v1x * v1x + v1y * v1y);
      eigenvec1 = [v1x / len1, v1y / len1];
      
      // Second eigenvector
      const v2x = lambda2 - covYY;
      const v2y = covXY;
      const len2 = Math.sqrt(v2x * v2x + v2y * v2y);
      eigenvec2 = [v2x / len2, v2y / len2];
    } else {
      eigenvec1 = [1, 0];
      eigenvec2 = [0, 1];
    }

    // Sort by eigenvalue (largest first)
    if (lambda1 > lambda2) {
      setEigenvalues([lambda1, lambda2]);
      setEigenvectors([eigenvec1, eigenvec2]);
    } else {
      setEigenvalues([lambda2, lambda1]);
      setEigenvectors([eigenvec2, eigenvec1]);
    }

    // Step 4: Project data onto principal components
    const projected = centered.map(p => {
      const pc1 = p.x * eigenvec1[0] + p.y * eigenvec1[1];
      const pc2 = p.x * eigenvec2[0] + p.y * eigenvec2[1];
      return { pc1, pc2, original: p };
    });

    setPrincipalComponents(projected);
  };

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 60;
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = 30;

    ctx.clearRect(0, 0, width, height);

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, centerY);
    ctx.lineTo(width - padding, centerY);
    ctx.moveTo(centerX, padding);
    ctx.lineTo(centerX, height - padding);
    ctx.stroke();

    // Draw data points
    ctx.fillStyle = '#ef4444';
    dataPoints.forEach(point => {
      const x = centerX + point.x * scale;
      const y = centerY - point.y * scale; // Flip Y axis
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Draw eigenvectors (principal components)
    if (eigenvectors.length === 2 && showPCA) {
      const [eigenvec1, eigenvec2] = eigenvectors;
      const [eigenval1, eigenval2] = eigenvalues;

      // First principal component (largest eigenvalue)
      const pc1Length = Math.sqrt(eigenval1) * scale * 2;
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(
        centerX + eigenvec1[0] * pc1Length,
        centerY - eigenvec1[1] * pc1Length
      );
      ctx.stroke();

      // Arrowhead for PC1
      const angle1 = Math.atan2(-eigenvec1[1], eigenvec1[0]);
      ctx.beginPath();
      ctx.moveTo(
        centerX + eigenvec1[0] * pc1Length,
        centerY - eigenvec1[1] * pc1Length
      );
      ctx.lineTo(
        centerX + eigenvec1[0] * pc1Length - 10 * Math.cos(angle1 - Math.PI / 6),
        centerY - eigenvec1[1] * pc1Length + 10 * Math.sin(angle1 - Math.PI / 6)
      );
      ctx.moveTo(
        centerX + eigenvec1[0] * pc1Length,
        centerY - eigenvec1[1] * pc1Length
      );
      ctx.lineTo(
        centerX + eigenvec1[0] * pc1Length - 10 * Math.cos(angle1 + Math.PI / 6),
        centerY - eigenvec1[1] * pc1Length + 10 * Math.sin(angle1 + Math.PI / 6)
      );
      ctx.stroke();

      // Second principal component
      const pc2Length = Math.sqrt(eigenval2) * scale * 2;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(
        centerX + eigenvec2[0] * pc2Length,
        centerY - eigenvec2[1] * pc2Length
      );
      ctx.stroke();

      // Arrowhead for PC2
      const angle2 = Math.atan2(-eigenvec2[1], eigenvec2[0]);
      ctx.beginPath();
      ctx.moveTo(
        centerX + eigenvec2[0] * pc2Length,
        centerY - eigenvec2[1] * pc2Length
      );
      ctx.lineTo(
        centerX + eigenvec2[0] * pc2Length - 10 * Math.cos(angle2 - Math.PI / 6),
        centerY - eigenvec2[1] * pc2Length + 10 * Math.sin(angle2 - Math.PI / 6)
      );
      ctx.moveTo(
        centerX + eigenvec2[0] * pc2Length,
        centerY - eigenvec2[1] * pc2Length
      );
      ctx.lineTo(
        centerX + eigenvec2[0] * pc2Length - 10 * Math.cos(angle2 + Math.PI / 6),
        centerY - eigenvec2[1] * pc2Length + 10 * Math.sin(angle2 + Math.PI / 6)
      );
      ctx.stroke();

      // Labels
      ctx.fillStyle = '#3b82f6';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(
        `PC1 (λ=${eigenval1.toFixed(2)})`,
        centerX + eigenvec1[0] * pc1Length + 10,
        centerY - eigenvec1[1] * pc1Length - 10
      );
      
      ctx.fillStyle = '#10b981';
      ctx.fillText(
        `PC2 (λ=${eigenval2.toFixed(2)})`,
        centerX + eigenvec2[0] * pc2Length + 10,
        centerY - eigenvec2[1] * pc2Length - 10
      );

      // Draw projected points if dimension reduction
      if (dimension === 1 && principalComponents.length > 0) {
        ctx.fillStyle = '#8b5cf6';
        principalComponents.forEach(proj => {
          const x = centerX + proj.pc1 * eigenvec1[0] * scale;
          const y = centerY - proj.pc1 * eigenvec1[1] * scale;
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    }

    // Labels
    ctx.fillStyle = '#000';
    ctx.font = '12px Arial';
    ctx.fillText('X', width - padding + 10, centerY + 20);
    ctx.fillText('Y', centerX - 30, padding - 10);
  };

  const codeExample = `# Principal Component Analysis (PCA)

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_manual(X):
    """
    Manual PCA Implementation
    
    Steps:
    1. Center the data (subtract mean)
    2. Calculate covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Project data onto principal components
    """
    # Step 1: Center data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Step 2: Covariance matrix
    # Cov = (1/n) X^T · X
    n = X_centered.shape[0]
    covariance_matrix = (1 / n) * np.dot(X_centered.T, X_centered)
    
    # Step 3: Eigenvalue decomposition
    # Cov · v = λ · v
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 4: Project data
    # Y = X · W
    # where W = matrix of eigenvectors
    X_projected = np.dot(X_centered, eigenvectors)
    
    return eigenvalues, eigenvectors, X_projected

# Example: 2D data
np.random.seed(42)
# Generate correlated data
X = np.random.randn(100, 2)
X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]  # Create correlation

# Manual PCA
eigenvalues, eigenvectors, X_projected = pca_manual(X)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Explained variance
variance_explained = eigenvalues / np.sum(eigenvalues)
print("Variance explained:", variance_explained)

# Using sklearn (for comparison)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Sklearn eigenvalues:", pca.explained_variance_)
print("Sklearn eigenvectors:", pca.components_)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Original Data')
axes[0].grid(True)

# Projected data (1D)
axes[1].scatter(X_projected[:, 0], np.zeros_like(X_projected[:, 0]), alpha=0.6)
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].set_title('PCA Projection (1D)')
axes[1].grid(True)

plt.tight_layout()
plt.show()`;

  const varianceExplained = eigenvalues.length > 0
    ? eigenvalues.map((val, idx) => ({
        pc: idx + 1,
        eigenvalue: val,
        variance: val / eigenvalues.reduce((a, b) => a + b, 0) * 100
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">PCA Visualization</h2>
        <p className="text-purple-800 text-sm">
          See how eigenvalues and eigenvectors from Linear Algebra are used for dimensionality reduction.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Dimensions to Keep
            </label>
            <select
              value={dimension}
              onChange={(e) => setDimension(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value={2}>2D (Original)</option>
              <option value={1}>1D (Reduced)</option>
            </select>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={() => setShowPCA(!showPCA)}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-semibold"
            >
              {showPCA ? 'Hide PCA' : 'Show PCA'}
            </button>
            <button
              onClick={generateSampleData}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
            >
              New Data
            </button>
          </div>
        </div>

        {/* Eigenvalues and Variance Explained */}
        {eigenvalues.length > 0 && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-900 mb-2">Eigenvalues & Variance Explained:</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              {varianceExplained.map((item, idx) => (
                <div key={idx} className="bg-white p-2 rounded">
                  <div><strong>PC{item.pc}:</strong> λ = {item.eigenvalue.toFixed(4)}</div>
                  <div className="text-gray-600">Variance: {item.variance.toFixed(2)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Canvas Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">PCA Visualization</h3>
        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={800}
            height={500}
            className="border-2 border-gray-300 rounded-lg"
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded-full"></div>
            <span>Data Points</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-500"></div>
            <span>PC1 (Largest Eigenvalue)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500"></div>
            <span>PC2 (Second Eigenvalue)</span>
          </div>
          {dimension === 1 && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
              <span>Projected (1D)</span>
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">PCA Implementation</h3>
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            customStyle={{ margin: 0, borderRadius: '0.5rem' }}
            showLineNumbers
          >
            {codeExample}
          </SyntaxHighlighter>
        </div>
      </div>

      {/* Mathematical Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Mathematical Foundations:</h3>
        <div className="space-y-2 text-blue-800 text-sm">
          <div>
            <strong>Linear Algebra:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Covariance Matrix: C = (1/n) X^T · X</li>
              <li>Eigenvalue Equation: C · v = λ · v</li>
              <li>Eigenvectors point in direction of maximum variance</li>
              <li>Eigenvalues represent variance along each direction</li>
            </ul>
          </div>
          <div>
            <strong>Dimensionality Reduction:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Project data: Y = X · W (where W = eigenvectors)</li>
              <li>Keep top k eigenvectors (largest eigenvalues)</li>
              <li>Preserves maximum variance in reduced dimensions</li>
            </ul>
          </div>
          <div>
            <strong>Statistics:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Covariance measures how features vary together</li>
              <li>Variance explained = eigenvalue / sum(eigenvalues)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

