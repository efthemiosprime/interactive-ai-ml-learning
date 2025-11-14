import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import * as ul from '../../utils/unsupervisedLearning';

export default function InteractiveDimensionalityReductionVisualization() {
  const canvasRef = useRef(null);
  const [algorithm, setAlgorithm] = useState('pca');
  const [data, setData] = useState(() => {
    // Generate 3D data
    const points = [];
    for (let i = 0; i < 50; i++) {
      const x = (Math.random() - 0.5) * 4;
      const y = (Math.random() - 0.5) * 4;
      const z = 0.5 * x + 0.3 * y + (Math.random() - 0.5) * 0.5; // Linear relationship
      points.push([x, y, z]);
    }
    return points;
  });
  const [pcaResult, setPcaResult] = useState(null);

  useEffect(() => {
    if (algorithm === 'pca') {
      const result = ul.pca(data, 2);
      setPcaResult(result);
    }
  }, [data, algorithm]);

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

    if (algorithm === 'pca' && pcaResult && pcaResult.transformed) {
      // Draw PCA transformed data
      const transformed = pcaResult.transformed;
      const xValues = transformed.map(p => p[0]);
      const yValues = transformed.map(p => p[1]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);
      const xRange = xMax - xMin || 1;
      const yRange = yMax - yMin || 1;

      const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
      const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const x = padding.left + (i / 5) * (width - padding.left - padding.right);
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
      }

      // Draw axes
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, height - padding.bottom);
      ctx.stroke();

      // Draw principal components as arrows
      if (pcaResult.components && pcaResult.components.length >= 2) {
        const pc1 = pcaResult.components[0];
        const pc2 = pcaResult.components[1];
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 50;

        // PC1
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + pc1[0] * scale, centerY - pc1[1] * scale);
        ctx.stroke();
        
        // PC2
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + pc2[0] * scale, centerY - pc2[1] * scale);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 12px sans-serif';
        ctx.fillText('PC1', centerX + pc1[0] * scale + 5, centerY - pc1[1] * scale - 5);
        ctx.fillStyle = '#3b82f6';
        ctx.fillText('PC2', centerX + pc2[0] * scale + 5, centerY - pc2[1] * scale - 5);
      }

      // Draw transformed points
      transformed.forEach(point => {
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(toScreenX(point[0]), toScreenY(point[1]), 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Draw title
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('PCA: 3D → 2D Projection', width / 2, 20);

      // Draw variance explained
      if (pcaResult.explainedVariance) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'left';
        const variance1 = (pcaResult.explainedVariance[0] * 100).toFixed(1);
        const variance2 = (pcaResult.explainedVariance[1] * 100).toFixed(1);
        ctx.fillText(`PC1 explains ${variance1}% variance`, padding.left, padding.top + 20);
        ctx.fillText(`PC2 explains ${variance2}% variance`, padding.left, padding.top + 35);
      }
    } else {
      // Draw original 3D data (projected to 2D)
      const xValues = data.map(p => p[0]);
      const yValues = data.map(p => p[1]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMin = Math.min(...yValues);
      const yMax = Math.max(...yValues);
      const xRange = xMax - xMin || 1;
      const yRange = yMax - yMin || 1;

      const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * (width - padding.left - padding.right);
      const toScreenY = (y) => padding.top + (height - padding.top - padding.bottom) - ((y - yMin) / yRange) * (height - padding.top - padding.bottom);

      data.forEach(point => {
        ctx.fillStyle = '#6b7280';
        ctx.beginPath();
        ctx.arc(toScreenX(point[0]), toScreenY(point[1]), 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Original 3D Data (projected)', width / 2, 20);
    }

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(algorithm === 'pca' && pcaResult ? 'Principal Component 1' : 'X', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(algorithm === 'pca' && pcaResult ? 'Principal Component 2' : 'Y', 0, 0);
    ctx.restore();
  }, [data, algorithm, pcaResult]);

  const generateNewData = () => {
    const points = [];
    for (let i = 0; i < 50; i++) {
      const x = (Math.random() - 0.5) * 4;
      const y = (Math.random() - 0.5) * 4;
      const z = 0.5 * x + 0.3 * y + (Math.random() - 0.5) * 0.5;
      points.push([x, y, z]);
    }
    setData(points);
  };

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border-2 border-teal-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-teal-800">
          💡 <strong>Interactive:</strong> See how PCA reduces 3D data to 2D while preserving maximum variance!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-teal-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Dimensionality Reduction Algorithm
          </label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500"
          >
            <option value="pca">PCA (Principal Component Analysis)</option>
            <option value="tsne" disabled>t-SNE (Coming Soon)</option>
            <option value="autoencoder" disabled>Autoencoders (Coming Soon)</option>
          </select>
        </div>

        <button
          onClick={generateNewData}
          className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Generate New Data
        </button>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Algorithm Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">PCA Explained:</h4>
        <ul className="text-sm text-gray-700 space-y-1">
          <li><strong>Goal:</strong> Find directions of maximum variance in high-dimensional data</li>
          <li><strong>Method:</strong> Uses eigenvalues and eigenvectors of covariance matrix</li>
          <li><strong>Result:</strong> Projects data onto principal components (orthogonal directions)</li>
          <li><strong>Use Case:</strong> Visualization, noise reduction, feature extraction</li>
        </ul>
      </div>
    </div>
  );
}

