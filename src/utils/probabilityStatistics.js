// Probability & Statistics utilities for AI/ML applications

import * as math from './math.js';

/**
 * Calculate descriptive statistics for a dataset
 */
export function calculateDescriptiveStats(data) {
  const mean = math.mean(data);
  const variance = math.variance(data);
  const stdDev = math.standardDeviation(data);
  
  const sorted = [...data].sort((a, b) => a - b);
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];
  
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;
  
  return {
    mean,
    median,
    variance,
    standardDeviation: stdDev,
    min,
    max,
    range,
    count: data.length
  };
}

/**
 * Calculate covariance matrix for multiple features
 */
export function covarianceMatrix(features) {
  const n = features.length;
  const m = features[0].length;
  const matrix = Array(m).fill(0).map(() => Array(m).fill(0));
  
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const featureI = features.map(row => row[i]);
      const featureJ = features.map(row => row[j]);
      matrix[i][j] = math.covariance(featureI, featureJ);
    }
  }
  
  return matrix;
}

/**
 * Calculate conditional probability with examples
 */
export function calculateConditionalProbability(pAAndB, pB) {
  return math.conditionalProbability(pAAndB, pB);
}

/**
 * Apply Bayes' theorem
 */
export function applyBayesTheorem(pBGivenA, pA, pB) {
  return math.bayesTheorem(pBGivenA, pA, pB);
}

/**
 * Generate samples from normal distribution
 */
export function sampleNormal(mean, stdDev, count = 100) {
  const samples = [];
  for (let i = 0; i < count; i++) {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    samples.push(z0 * stdDev + mean);
  }
  return samples;
}

/**
 * Generate samples from Bernoulli distribution
 */
export function sampleBernoulli(p, count = 100) {
  const samples = [];
  for (let i = 0; i < count; i++) {
    samples.push(Math.random() < p ? 1 : 0);
  }
  return samples;
}

/**
 * Calculate probability density for normal distribution
 */
export function normalDistributionPDF(x, mean, stdDev) {
  return math.normalPDF(x, mean, stdDev);
}

/**
 * Calculate probability mass for Bernoulli distribution
 */
export function bernoulliDistributionPMF(x, p) {
  return math.bernoulliPMF(x, p);
}

/**
 * Calculate z-score (standardization)
 */
export function zScore(value, mean, stdDev) {
  if (stdDev === 0) return 0;
  return (value - mean) / stdDev;
}

/**
 * Standardize a dataset (z-score normalization)
 */
export function standardize(data) {
  const mean = math.mean(data);
  const stdDev = math.standardDeviation(data);
  
  if (stdDev === 0) return data.map(() => 0);
  
  return data.map(value => zScore(value, mean, stdDev));
}

