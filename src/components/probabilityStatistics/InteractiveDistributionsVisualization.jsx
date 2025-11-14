import React, { useRef, useEffect, useState } from 'react';
import * as stats from '../../utils/probabilityStatistics';

export default function InteractiveDistributionsVisualization() {
  const canvasRef = useRef(null);
  const [distributionType, setDistributionType] = useState('normal');
  const [normalMean, setNormalMean] = useState(0);
  const [normalStdDev, setNormalStdDev] = useState(1);
  const [bernoulliP, setBernoulliP] = useState(0.5);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 600;
    const displayHeight = rect.height > 0 ? rect.height : 400;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (distributionType === 'normal') {
      // Draw Normal Distribution
      const xMin = normalMean - 4 * normalStdDev;
      const xMax = normalMean + 4 * normalStdDev;
      const xRange = xMax - xMin;
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;

      const toScreenX = (x) => padding.left + ((x - xMin) / xRange) * plotWidth;
      const toScreenY = (y) => padding.top + plotHeight - (y * plotHeight * 2);

      // Calculate PDF values
      const points = [];
      let maxPDF = 0;
      for (let x = xMin; x <= xMax; x += 0.1) {
        const pdf = stats.normalDistributionPDF(x, normalMean, normalStdDev);
        points.push({ x, pdf });
        maxPDF = Math.max(maxPDF, pdf);
      }

      // Normalize for display
      const scaleFactor = 0.9 / maxPDF;
      points.forEach(p => p.pdf *= scaleFactor);

      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const x = padding.left + (i / 5) * plotWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
      }

      // Draw bell curve
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      let firstPoint = true;
      points.forEach(point => {
        const screenX = toScreenX(point.x);
        const screenY = toScreenY(point.pdf);
        if (firstPoint) {
          ctx.moveTo(screenX, screenY);
          firstPoint = false;
        } else {
          ctx.lineTo(screenX, screenY);
        }
      });
      ctx.stroke();

      // Fill area under curve
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.beginPath();
      ctx.moveTo(toScreenX(xMin), height - padding.bottom);
      points.forEach(point => {
        ctx.lineTo(toScreenX(point.x), toScreenY(point.pdf));
      });
      ctx.lineTo(toScreenX(xMax), height - padding.bottom);
      ctx.closePath();
      ctx.fill();

      // Draw mean line
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      const meanX = toScreenX(normalMean);
      ctx.beginPath();
      ctx.moveTo(meanX, padding.top);
      ctx.lineTo(meanX, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw std dev lines
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      const stdDev1X = toScreenX(normalMean + normalStdDev);
      const stdDev2X = toScreenX(normalMean - normalStdDev);
      ctx.beginPath();
      ctx.moveTo(stdDev1X, padding.top);
      ctx.lineTo(stdDev1X, height - padding.bottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(stdDev2X, padding.top);
      ctx.lineTo(stdDev2X, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw axis labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      for (let i = 0; i <= 5; i++) {
        const value = xMin + (i / 5) * xRange;
        const x = padding.left + (i / 5) * plotWidth;
        ctx.fillText(value.toFixed(1), x, height - padding.bottom + 20);
      }

      // Draw title
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Normal Distribution (μ=${normalMean.toFixed(1)}, σ=${normalStdDev.toFixed(1)})`, width / 2, 20);

    } else if (distributionType === 'bernoulli') {
      // Draw Bernoulli Distribution
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const barWidth = plotWidth / 3;
      const centerX = width / 2;

      // Draw bars
      const p0 = 1 - bernoulliP;
      const maxProb = Math.max(bernoulliP, p0);

      // P(X=0)
      const bar0Height = (p0 / maxProb) * plotHeight * 0.8;
      const bar0X = centerX - barWidth;
      ctx.fillStyle = '#8b5cf6';
      ctx.fillRect(bar0X, height - padding.bottom - bar0Height, barWidth - 10, bar0Height);

      // P(X=1)
      const bar1Height = (bernoulliP / maxProb) * plotHeight * 0.8;
      const bar1X = centerX + 10;
      ctx.fillStyle = '#6366f1';
      ctx.fillRect(bar1X, height - padding.bottom - bar1Height, barWidth - 10, bar1Height);

      // Draw labels
      ctx.fillStyle = '#6b7280';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('0', bar0X + (barWidth - 10) / 2, height - padding.bottom + 20);
      ctx.fillText('1', bar1X + (barWidth - 10) / 2, height - padding.bottom + 20);

      // Draw probability labels
      ctx.fillStyle = '#8b5cf6';
      ctx.font = 'bold 16px sans-serif';
      ctx.fillText(p0.toFixed(2), bar0X + (barWidth - 10) / 2, height - padding.bottom - bar0Height - 10);
      
      ctx.fillStyle = '#6366f1';
      ctx.fillText(bernoulliP.toFixed(2), bar1X + (barWidth - 10) / 2, height - padding.bottom - bar1Height - 10);

      // Draw title
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(`Bernoulli Distribution (p=${bernoulliP.toFixed(2)})`, width / 2, 20);

      // Draw axis
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();
  }, [distributionType, normalMean, normalStdDev, bernoulliP]);

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Adjust the parameters below to see how distributions change!
        </p>
      </div>

      {/* Distribution Type Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-blue-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Distribution Type
        </label>
        <select
          value={distributionType}
          onChange={(e) => setDistributionType(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="normal">Normal Distribution</option>
          <option value="bernoulli">Bernoulli Distribution</option>
        </select>
      </div>

      {/* Parameters */}
      {distributionType === 'normal' && (
        <div className="bg-white rounded-lg p-4 border-2 border-purple-200 space-y-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Mean (μ): {normalMean.toFixed(1)}
            </label>
            <input
              type="range"
              min="-3"
              max="3"
              step="0.1"
              value={normalMean}
              onChange={(e) => setNormalMean(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Standard Deviation (σ): {normalStdDev.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.5"
              max="3"
              step="0.1"
              value={normalStdDev}
              onChange={(e) => setNormalStdDev(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      )}

      {distributionType === 'bernoulli' && (
        <div className="bg-white rounded-lg p-4 border-2 border-purple-200">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Probability of Success (p): {bernoulliP.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={bernoulliP}
              onChange={(e) => setBernoulliP(Number(e.target.value))}
              className="w-full"
            />
            <div className="mt-2 text-sm text-gray-600">
              P(X=0) = {1 - bernoulliP.toFixed(2)}, P(X=1) = {bernoulliP.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '400px' }}
      />
    </div>
  );
}

