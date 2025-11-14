// Unsupervised Learning utilities for AI/ML applications

import * as math from './math.js';
import * as linearAlgebra from './linearAlgebra.js';

/**
 * Calculate Euclidean distance between two points
 */
export function euclideanDistance(point1, point2) {
  if (point1.length !== point2.length) {
    throw new Error('Points must have the same dimension');
  }
  let sum = 0;
  for (let i = 0; i < point1.length; i++) {
    sum += Math.pow(point1[i] - point2[i], 2);
  }
  return Math.sqrt(sum);
}

/**
 * Calculate Manhattan distance (L1 distance)
 */
export function manhattanDistance(point1, point2) {
  if (point1.length !== point2.length) {
    throw new Error('Points must have the same dimension');
  }
  let sum = 0;
  for (let i = 0; i < point1.length; i++) {
    sum += Math.abs(point1[i] - point2[i]);
  }
  return sum;
}

/**
 * Calculate cosine similarity between two vectors
 */
export function cosineSimilarity(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same dimension');
  }
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  for (let i = 0; i < vector1.length; i++) {
    dotProduct += vector1[i] * vector2[i];
    norm1 += vector1[i] * vector1[i];
    norm2 += vector2[i] * vector2[i];
  }
  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * K-means clustering algorithm
 */
export function kMeans(data, k, maxIterations = 100) {
  if (data.length === 0 || k <= 0 || k > data.length) {
    return { centroids: [], clusters: [], iterations: 0 };
  }

  const dimensions = data[0].length;
  
  // Initialize centroids randomly
  let centroids = [];
  for (let i = 0; i < k; i++) {
    const randomIndex = Math.floor(Math.random() * data.length);
    centroids.push([...data[randomIndex]]);
  }

  let clusters = [];
  let iterations = 0;
  let converged = false;

  while (iterations < maxIterations && !converged) {
    // Assign points to nearest centroid
    clusters = data.map(point => {
      let minDistance = Infinity;
      let nearestCentroid = 0;
      for (let i = 0; i < centroids.length; i++) {
        const distance = euclideanDistance(point, centroids[i]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCentroid = i;
        }
      }
      return nearestCentroid;
    });

    // Update centroids
    const newCentroids = [];
    let hasChanged = false;

    for (let i = 0; i < k; i++) {
      const clusterPoints = data.filter((_, idx) => clusters[idx] === i);
      if (clusterPoints.length === 0) {
        newCentroids.push([...centroids[i]]);
        continue;
      }

      const newCentroid = [];
      for (let d = 0; d < dimensions; d++) {
        const sum = clusterPoints.reduce((acc, point) => acc + point[d], 0);
        newCentroid.push(sum / clusterPoints.length);
      }

      // Check if centroid changed
      if (euclideanDistance(newCentroid, centroids[i]) > 0.001) {
        hasChanged = true;
      }
      newCentroids.push(newCentroid);
    }

    converged = !hasChanged;
    centroids = newCentroids;
    iterations++;
  }

  return { centroids, clusters, iterations };
}

/**
 * Calculate Principal Component Analysis (PCA)
 * Uses simplified eigenvalue decomposition for any matrix size
 */
export function pca(data, numComponents = 2) {
  if (data.length === 0) return { components: [], explainedVariance: [], transformed: [] };

  const n = data.length;
  const m = data[0].length;

  // Center the data (subtract mean)
  const mean = [];
  for (let j = 0; j < m; j++) {
    mean.push(data.reduce((sum, row) => sum + row[j], 0) / n);
  }

  const centeredData = data.map(row => row.map((val, j) => val - mean[j]));

  // Calculate covariance matrix
  const covariance = [];
  for (let i = 0; i < m; i++) {
    covariance[i] = [];
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centeredData[k][i] * centeredData[k][j];
      }
      covariance[i][j] = sum / (n - 1);
    }
  }

  // Simplified eigenvalue decomposition using power iteration
  // For 2x2 matrices, use the existing function
  let eigenPairs = [];
  
  if (m === 2) {
    try {
      const eigenDecomp = linearAlgebra.calculateEigenDecomposition(covariance);
      if (!eigenDecomp.isComplex && eigenDecomp.eigenvectors) {
        eigenPairs = eigenDecomp.eigenvalues.map((val, idx) => ({
          eigenvalue: typeof val === 'object' ? val.real : val,
          eigenvector: [
            eigenDecomp.eigenvectors[idx]?.eigenvector?.x || 0,
            eigenDecomp.eigenvectors[idx]?.eigenvector?.y || 0
          ]
        })).sort((a, b) => b.eigenvalue - a.eigenvalue);
      }
    } catch (e) {
      // Fall through to power iteration
    }
  }

  // If 2x2 failed or matrix is larger, use power iteration
  if (eigenPairs.length === 0) {
    // Power iteration for largest eigenvalue/eigenvector
    for (let comp = 0; comp < Math.min(numComponents, m); comp++) {
      // Start with random vector
      let v = [];
      for (let i = 0; i < m; i++) {
        v.push(Math.random() - 0.5);
      }
      
      // Normalize
      let norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
      v = v.map(x => x / norm);
      
      // Power iteration
      for (let iter = 0; iter < 50; iter++) {
        // Multiply covariance by v
        let Av = [];
        for (let i = 0; i < m; i++) {
          let sum = 0;
          for (let j = 0; j < m; j++) {
            sum += covariance[i][j] * v[j];
          }
          Av.push(sum);
        }
        
        // Normalize
        norm = Math.sqrt(Av.reduce((sum, x) => sum + x * x, 0));
        if (norm < 1e-10) break;
        v = Av.map(x => x / norm);
      }
      
      // Calculate eigenvalue (Rayleigh quotient)
      let Av = [];
      for (let i = 0; i < m; i++) {
        let sum = 0;
        for (let j = 0; j < m; j++) {
          sum += covariance[i][j] * v[j];
        }
        Av.push(sum);
      }
      const eigenvalue = v.reduce((sum, vi, i) => sum + vi * Av[i], 0);
      
      eigenPairs.push({ eigenvalue, eigenvector: v });
      
      // Deflate matrix for next component
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          covariance[i][j] -= eigenvalue * v[i] * v[j];
        }
      }
    }
    
    // Sort by eigenvalue
    eigenPairs.sort((a, b) => b.eigenvalue - a.eigenvalue);
  }

  // Select top components
  const components = eigenPairs.slice(0, numComponents).map(pair => pair.eigenvector);

  const totalVariance = eigenPairs.reduce((sum, p) => sum + Math.max(0, p.eigenvalue), 0);
  const explainedVariance = eigenPairs.slice(0, numComponents).map(p => 
    totalVariance > 0 ? Math.max(0, p.eigenvalue) / totalVariance : 0
  );

  // Transform data (project onto principal components)
  const transformed = centeredData.map(point => {
    return components.map(component => {
      return point.reduce((sum, val, idx) => sum + val * component[idx], 0);
    });
  });

  return { 
    components, 
    explainedVariance, 
    transformed, 
    eigenvalues: eigenPairs.slice(0, numComponents).map(p => Math.max(0, p.eigenvalue)),
    mean
  };
}

/**
 * Generate sample 2D data for clustering visualization
 */
export function generateClusteringData(numPoints = 50, numClusters = 3, spread = 1.0) {
  const data = [];
  const clusterCenters = [];
  
  // Generate cluster centers
  for (let i = 0; i < numClusters; i++) {
    const angle = (i / numClusters) * 2 * Math.PI;
    const radius = 2;
    clusterCenters.push([
      Math.cos(angle) * radius,
      Math.sin(angle) * radius
    ]);
  }

  // Generate points around each center
  const pointsPerCluster = Math.floor(numPoints / numClusters);
  for (let c = 0; c < numClusters; c++) {
    for (let i = 0; i < pointsPerCluster; i++) {
      const x = clusterCenters[c][0] + (Math.random() - 0.5) * spread * 2;
      const y = clusterCenters[c][1] + (Math.random() - 0.5) * spread * 2;
      data.push([x, y]);
    }
  }

  return data;
}

/**
 * Generate sample data with anomalies
 */
export function generateAnomalyData(numNormal = 100, numAnomalies = 5) {
  const data = [];
  
  // Generate normal data (clustered)
  for (let i = 0; i < numNormal; i++) {
    const x = (Math.random() - 0.5) * 4;
    const y = (Math.random() - 0.5) * 4;
    data.push({ point: [x, y], isAnomaly: false });
  }

  // Generate anomalies (outliers)
  for (let i = 0; i < numAnomalies; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const radius = 5 + Math.random() * 2;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    data.push({ point: [x, y], isAnomaly: true });
  }

  return data;
}

/**
 * Simple DBSCAN implementation (simplified for visualization)
 */
export function dbscan(data, eps = 1.0, minPts = 3) {
  const n = data.length;
  const visited = new Array(n).fill(false);
  const clusters = new Array(n).fill(-1); // -1 = noise
  let clusterId = 0;

  const getNeighbors = (pointIdx) => {
    const neighbors = [];
    for (let i = 0; i < n; i++) {
      if (i !== pointIdx && euclideanDistance(data[pointIdx], data[i]) <= eps) {
        neighbors.push(i);
      }
    }
    return neighbors;
  };

  const expandCluster = (pointIdx, neighbors, clusterId) => {
    clusters[pointIdx] = clusterId;
    let i = 0;
    while (i < neighbors.length) {
      const neighborIdx = neighbors[i];
      if (!visited[neighborIdx]) {
        visited[neighborIdx] = true;
        const neighborNeighbors = getNeighbors(neighborIdx);
        if (neighborNeighbors.length >= minPts) {
          neighbors.push(...neighborNeighbors);
        }
      }
      if (clusters[neighborIdx] === -1) {
        clusters[neighborIdx] = clusterId;
      }
      i++;
    }
  };

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;

    const neighbors = getNeighbors(i);
    if (neighbors.length < minPts) {
      clusters[i] = -1; // Noise
    } else {
      expandCluster(i, neighbors, clusterId);
      clusterId++;
    }
  }

  return { clusters, numClusters: clusterId };
}

