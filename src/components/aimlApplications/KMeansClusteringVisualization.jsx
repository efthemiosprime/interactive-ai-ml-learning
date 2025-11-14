import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function KMeansClusteringVisualization() {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [centroids, setCentroids] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [k, setK] = useState(3);
  const [isRunning, setIsRunning] = useState(false);
  const [iteration, setIteration] = useState(0);
  const [showDistances, setShowDistances] = useState(false);
  const [colors] = useState(['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']);

  useEffect(() => {
    generateSampleData();
  }, []);

  useEffect(() => {
    if (dataPoints.length > 0) {
      // Only initialize if centroids are empty or k changed
      if (centroids.length === 0 || centroids.length !== k) {
        initializeCentroids();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, dataPoints.length]);

  useEffect(() => {
    drawVisualization();
  }, [dataPoints, centroids, clusters, showDistances, iteration]);

  const generateSampleData = () => {
    const points = [];
    
    // Generate 3 clusters of data
    const clusterCenters = [
      { x: 150, y: 150 },
      { x: 400, y: 200 },
      { x: 300, y: 400 }
    ];
    
    clusterCenters.forEach(center => {
      for (let i = 0; i < 20; i++) {
        const angle = Math.random() * 2 * Math.PI;
        const radius = Math.random() * 60 + 20;
        points.push({
          x: center.x + radius * Math.cos(angle),
          y: center.y + radius * Math.sin(angle),
          cluster: -1
        });
      }
    });
    
    setDataPoints(points);
    setIteration(0);
    setClusters([]);
  };

  const initializeCentroids = () => {
    if (dataPoints.length === 0) return;
    
    // K-means++ initialization (better than random)
    const newCentroids = [];
    const usedIndices = new Set();
    
    // First centroid: random point
    const firstIdx = Math.floor(Math.random() * dataPoints.length);
    newCentroids.push({ ...dataPoints[firstIdx], id: 0 });
    usedIndices.add(firstIdx);
    
    // Subsequent centroids: farthest from existing centroids
    for (let i = 1; i < k; i++) {
      let maxDist = -1;
      let farthestIdx = -1;
      
      dataPoints.forEach((point, idx) => {
        if (usedIndices.has(idx)) return;
        
        // Find minimum distance to any existing centroid
        let minDist = Infinity;
        newCentroids.forEach(centroid => {
          const dist = euclideanDistance(point, centroid);
          if (dist < minDist) minDist = dist;
        });
        
        if (minDist > maxDist) {
          maxDist = minDist;
          farthestIdx = idx;
        }
      });
      
      if (farthestIdx >= 0) {
        newCentroids.push({ ...dataPoints[farthestIdx], id: i });
        usedIndices.add(farthestIdx);
      }
    }
    
    setCentroids(newCentroids);
    setIteration(0);
    setClusters([]);
  };

  const euclideanDistance = (p1, p2) => {
    // Linear Algebra: Euclidean distance formula
    // d = √((x₁ - x₂)² + (y₁ - y₂)²)
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  };

  const assignClusters = () => {
    // Assign each point to nearest centroid
    const newClusters = dataPoints.map(point => {
      let minDist = Infinity;
      let nearestCentroidId = -1;
      
      centroids.forEach(centroid => {
        const dist = euclideanDistance(point, centroid);
        if (dist < minDist) {
          minDist = dist;
          nearestCentroidId = centroid.id;
        }
      });
      
      return {
        ...point,
        cluster: nearestCentroidId,
        distance: minDist
      };
    });
    
    setClusters(newClusters);
    return newClusters;
  };

  const updateCentroids = (assignedClusters) => {
    // Calculate new centroids as mean of assigned points
    // Statistics: Mean calculation
    const newCentroids = centroids.map(centroid => {
      const clusterPoints = assignedClusters.filter(p => p.cluster === centroid.id);
      
      if (clusterPoints.length === 0) {
        return centroid; // Keep old centroid if no points assigned
      }
      
      // Calculate mean (centroid)
      const meanX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
      const meanY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
      
      return {
        ...centroid,
        x: meanX,
        y: meanY
      };
    });
    
    setCentroids(newCentroids);
    return newCentroids;
  };

  const calculateWCSS = () => {
    // Within-Cluster Sum of Squares (variance measure)
    // Statistics: Variance calculation
    let totalWCSS = 0;
    
    centroids.forEach(centroid => {
      const clusterPoints = clusters.filter(p => p.cluster === centroid.id);
      clusterPoints.forEach(point => {
        const dist = euclideanDistance(point, centroid);
        totalWCSS += dist * dist; // Squared distance
      });
    });
    
    return totalWCSS;
  };

  const runKMeans = async () => {
    if (dataPoints.length === 0) {
      return;
    }
    
    setIsRunning(true);
    setIteration(0);
    
    // Get current centroids - initialize if needed
    let currentCentroids;
    if (centroids.length === 0 || centroids.length !== k) {
      // Initialize centroids synchronously
      const newCentroids = [];
      const usedIndices = new Set();
      
      const firstIdx = Math.floor(Math.random() * dataPoints.length);
      newCentroids.push({ ...dataPoints[firstIdx], id: 0 });
      usedIndices.add(firstIdx);
      
      for (let i = 1; i < k; i++) {
        let maxDist = -1;
        let farthestIdx = -1;
        
        dataPoints.forEach((point, idx) => {
          if (usedIndices.has(idx)) return;
          
          let minDist = Infinity;
          newCentroids.forEach(centroid => {
            const dist = euclideanDistance(point, centroid);
            if (dist < minDist) minDist = dist;
          });
          
          if (minDist > maxDist) {
            maxDist = minDist;
            farthestIdx = idx;
          }
        });
        
        if (farthestIdx >= 0) {
          newCentroids.push({ ...dataPoints[farthestIdx], id: i });
          usedIndices.add(farthestIdx);
        }
      }
      
      setCentroids(newCentroids);
      currentCentroids = newCentroids.map(c => ({ ...c }));
    } else {
      currentCentroids = centroids.map(c => ({ ...c }));
    }
    let converged = false;
    let maxIterations = 20;
    
    for (let iter = 0; iter < maxIterations && !converged; iter++) {
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Step 1: Assign clusters
      const assignedClusters = dataPoints.map(point => {
        let minDist = Infinity;
        let nearestCentroidId = -1;
        
        currentCentroids.forEach(centroid => {
          const dist = euclideanDistance(point, centroid);
          if (dist < minDist) {
            minDist = dist;
            nearestCentroidId = centroid.id;
          }
        });
        
        return {
          ...point,
          cluster: nearestCentroidId,
          distance: minDist
        };
      });
      
      setClusters(assignedClusters);
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Step 2: Update centroids
      const newCentroids = currentCentroids.map(centroid => {
        const clusterPoints = assignedClusters.filter(p => p.cluster === centroid.id);
        
        if (clusterPoints.length === 0) {
          return { ...centroid };
        }
        
        const meanX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
        const meanY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
        
        return {
          ...centroid,
          x: meanX,
          y: meanY
        };
      });
      
      // Check convergence (centroids didn't move much)
      converged = currentCentroids.every((oldCentroid, idx) => {
        const newCentroid = newCentroids[idx];
        const dist = euclideanDistance(oldCentroid, newCentroid);
        return dist < 1; // Threshold for convergence
      });
      
      currentCentroids = newCentroids;
      setCentroids([...newCentroids]);
      setIteration(iter + 1);
      
      if (converged) {
        break;
      }
    }
    
    setIsRunning(false);
  };

  const drawVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);

    // Draw cluster assignments
    const pointsToDraw = clusters.length > 0 ? clusters : dataPoints;
    
    pointsToDraw.forEach(point => {
      const clusterId = point.cluster >= 0 ? point.cluster : -1;
      const color = clusterId >= 0 ? colors[clusterId % colors.length] : '#9ca3af';
      
      // Draw point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw distance line if enabled
      if (showDistances && clusterId >= 0 && centroids[clusterId]) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(point.x, point.y);
        ctx.lineTo(centroids[clusterId].x, centroids[clusterId].y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw centroids
    centroids.forEach((centroid, idx) => {
      const color = colors[idx % colors.length];
      
      // Draw centroid circle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(centroid.x, centroid.y, 12, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw centroid border
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw centroid label
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('C' + idx, centroid.x, centroid.y + 4);
    });
  };

  const codeExample = `# K-Means Clustering Algorithm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_manual(X, k, max_iterations=100):
    """
    K-Means Clustering Algorithm
    
    Steps:
    1. Initialize k centroids (K-means++ initialization)
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence
    
    Mathematical Foundations:
    - Linear Algebra: Euclidean distance d = √(Σ(xᵢ - cᵢ)²)
    - Statistics: Mean calculation for centroids
    - Optimization: Minimize Within-Cluster Sum of Squares (WCSS)
    """
    n_samples, n_features = X.shape
    
    # Step 1: Initialize centroids (K-means++)
    centroids = np.zeros((k, n_features))
    
    # First centroid: random point
    centroids[0] = X[np.random.randint(n_samples)]
    
    # Subsequent centroids: farthest from existing
    for i in range(1, k):
        distances = np.array([
            min([np.linalg.norm(x - c)**2 for c in centroids[:i]])
            for x in X
        ])
        probabilities = distances / distances.sum()
        cumulative_probs = probabilities.cumsum()
        r = np.random.rand()
        centroids[i] = X[np.where(cumulative_probs >= r)[0][0]]
    
    # Iterate
    for iteration in range(max_iterations):
        # Step 2: Assign clusters (Euclidean distance)
        # Linear Algebra: Distance calculation
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Step 3: Update centroids (Mean calculation)
        # Statistics: Mean of points in each cluster
        new_centroids = np.array([
            X[labels == i].mean(axis=0)
            for i in range(k)
        ])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Calculate WCSS (Within-Cluster Sum of Squares)
def calculate_wcss(X, centroids, labels):
    """
    WCSS = Σ Σ ||x - c||²
    
    where:
    - x = data point
    - c = centroid of cluster
    - Measures variance within clusters
    """
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroids[i])**2)
    return wcss

# Example usage
np.random.seed(42)

# Generate sample data (3 clusters)
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [-2, -2]
cluster3 = np.random.randn(50, 2) + [2, -2]
X = np.vstack([cluster1, cluster2, cluster3])

# Run K-Means
k = 3
centroids, labels = kmeans_manual(X, k)

print(f"Centroids:")
print(centroids)
print(f"WCSS: {calculate_wcss(X, centroids, labels):.2f}")

# Visualize
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
    plt.scatter(centroids[i, 0], centroids[i, 1], 
                c='black', marker='x', s=200, linewidths=3, label='Centroid' if i == 0 else '')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (k=3)')
plt.legend()
plt.grid(True)
plt.show()

# Using sklearn (for comparison)
kmeans_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_sklearn.fit(X)
print(f"Sklearn WCSS: {kmeans_sklearn.inertia_:.2f}")`;

  const wcss = clusters.length > 0 ? calculateWCSS() : 0;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">K-Means Clustering Visualization</h2>
        <p className="text-purple-800 text-sm">
          Watch how K-Means groups data points into clusters using distance metrics and centroid updates.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Number of Clusters (K): {k}
            </label>
            <input
              type="range"
              min="2"
              max="6"
              step="1"
              value={k}
              onChange={(e) => {
                setK(parseInt(e.target.value));
                setIsRunning(false);
              }}
              disabled={isRunning}
              className="w-full"
            />
            <div className="text-xs text-gray-500 mt-1">
              More clusters = finer grouping
            </div>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={() => {
                if (centroids.length === 0 && dataPoints.length > 0) {
                  initializeCentroids();
                  setTimeout(() => runKMeans(), 100);
                } else {
                  runKMeans();
                }
              }}
              disabled={isRunning || dataPoints.length === 0}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isRunning ? 'Running...' : 'Run K-Means'}
            </button>
            <button
              onClick={() => {
                generateSampleData();
              }}
              disabled={isRunning}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              New Data
            </button>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showDistances}
              onChange={(e) => setShowDistances(e.target.checked)}
              className="w-4 h-4"
            />
            <label className="text-sm font-semibold text-gray-700">Show Distances</label>
          </div>
        </div>

        {/* Status */}
        {(iteration > 0 || wcss > 0) && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <strong>Iteration:</strong> {iteration}
              </div>
              <div>
                <strong>WCSS:</strong> {wcss.toFixed(2)}
              </div>
              <div>
                <strong>Clusters:</strong> {k}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Canvas Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">K-Means Clustering Visualization</h3>
        <div className="flex justify-center">
          <canvas
            ref={canvasRef}
            width={600}
            height={600}
            className="border-2 border-gray-300 rounded-lg"
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-400 rounded-full"></div>
            <span>Data Points</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-black rounded-full bg-blue-500"></div>
            <span>Centroids (C0, C1, ...)</span>
          </div>
          {showDistances && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-dashed border-2 border-blue-500"></div>
              <span>Distance to Centroid</span>
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">K-Means Implementation</h3>
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
            <strong>Linear Algebra - Euclidean Distance:</strong>
            <p className="ml-4 mt-1">d(x, c) = √(Σ(xᵢ - cᵢ)²)</p>
            <p className="ml-4 text-xs text-blue-600">Distance between point x and centroid c</p>
          </div>
          <div>
            <strong>Statistics - Centroid Update:</strong>
            <p className="ml-4 mt-1">c = (1/n) Σ xᵢ</p>
            <p className="ml-4 text-xs text-blue-600">Mean of all points in cluster</p>
          </div>
          <div>
            <strong>Optimization - WCSS (Within-Cluster Sum of Squares):</strong>
            <p className="ml-4 mt-1">WCSS = Σ Σ ||x - c||²</p>
            <p className="ml-4 text-xs text-blue-600">Minimize variance within clusters</p>
          </div>
          <div>
            <strong>Algorithm Steps:</strong>
            <ol className="ml-4 mt-1 list-decimal list-inside">
              <li>Initialize k centroids (K-means++)</li>
              <li>Assign each point to nearest centroid (distance calculation)</li>
              <li>Update centroids as mean of assigned points</li>
              <li>Repeat until convergence</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}

