import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw, Play } from 'lucide-react';
import * as ul from '../../utils/unsupervisedLearning';

export default function InteractiveClusteringVisualization() {
  const canvasRef = useRef(null);
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [k, setK] = useState(3);
  const [data, setData] = useState(() => ul.generateClusteringData(60, 3, 1.0));
  const [clusters, setClusters] = useState(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

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

    // Find data bounds
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

    // Draw data points
    if (clusters && clusters.clusters) {
      data.forEach((point, i) => {
        const clusterId = clusters.clusters[i];
        const color = clusterId >= 0 ? colors[clusterId % colors.length] : '#9ca3af';
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(toScreenX(point[0]), toScreenY(point[1]), 5, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Draw centroids (for K-means)
      if (algorithm === 'kmeans' && clusters.centroids) {
        clusters.centroids.forEach((centroid, i) => {
          ctx.fillStyle = colors[i % colors.length];
          ctx.strokeStyle = '#1f2937';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(toScreenX(centroid[0]), toScreenY(centroid[1]), 10, 0, 2 * Math.PI);
          ctx.fill();
          ctx.stroke();
          
          // Draw X mark
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(toScreenX(centroid[0]) - 5, toScreenY(centroid[1]) - 5);
          ctx.lineTo(toScreenX(centroid[0]) + 5, toScreenY(centroid[1]) + 5);
          ctx.moveTo(toScreenX(centroid[0]) - 5, toScreenY(centroid[1]) + 5);
          ctx.lineTo(toScreenX(centroid[0]) + 5, toScreenY(centroid[1]) - 5);
          ctx.stroke();
        });
      }
    } else {
      // Draw unclustered data
      data.forEach(point => {
        ctx.fillStyle = '#6b7280';
        ctx.beginPath();
        ctx.arc(toScreenX(point[0]), toScreenY(point[1]), 5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const algorithmNames = {
      kmeans: 'K-Means Clustering',
      hierarchical: 'Hierarchical Clustering',
      dbscan: 'DBSCAN Clustering'
    };
    ctx.fillText(algorithmNames[algorithm] || 'Clustering', width / 2, 20);

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('X', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Y', 0, 0);
    ctx.restore();
  }, [data, clusters, algorithm]);

  const runClustering = () => {
    setIsAnimating(true);
    
    setTimeout(() => {
      let result;
      if (algorithm === 'kmeans') {
        result = ul.kMeans(data, k);
      } else if (algorithm === 'dbscan') {
        result = ul.dbscan(data, 1.0, 3);
      } else {
        // Hierarchical clustering (simplified - using K-means for visualization)
        result = ul.kMeans(data, k);
      }
      
      setClusters(result);
      setIsAnimating(false);
    }, 100);
  };

  const generateNewData = () => {
    const newData = ul.generateClusteringData(60, 3, 1.0);
    setData(newData);
    setClusters(null);
  };

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border-2 border-teal-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-teal-800">
          💡 <strong>Interactive:</strong> Select an algorithm and click "Run Clustering" to see how data points are grouped!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-teal-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Clustering Algorithm
          </label>
          <select
            value={algorithm}
            onChange={(e) => {
              setAlgorithm(e.target.value);
              setClusters(null);
            }}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500"
          >
            <option value="kmeans">K-Means</option>
            <option value="hierarchical">Hierarchical Clustering</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </div>

        {algorithm === 'kmeans' && (
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Number of Clusters (k): {k}
            </label>
            <input
              type="range"
              min="2"
              max="6"
              step="1"
              value={k}
              onChange={(e) => {
                setK(Number(e.target.value));
                setClusters(null);
              }}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>2</span>
              <span>3</span>
              <span>4</span>
              <span>5</span>
              <span>6</span>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={runClustering}
            disabled={isAnimating}
            className="flex-1 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Play className="w-4 h-4" />
            Run Clustering
          </button>
          <button
            onClick={generateNewData}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            New Data
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Algorithm Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Algorithm Details:</h4>
        <p className="text-sm text-gray-700">
          {algorithm === 'kmeans' && 'K-Means partitions data into k clusters by minimizing within-cluster variance. Centroids are iteratively updated until convergence.'}
          {algorithm === 'hierarchical' && 'Hierarchical clustering builds a tree of clusters. Can be agglomerative (bottom-up) or divisive (top-down).'}
          {algorithm === 'dbscan' && 'DBSCAN groups points that are closely packed together, marking outliers as noise. Does not require specifying number of clusters.'}
        </p>
      </div>
    </div>
  );
}

