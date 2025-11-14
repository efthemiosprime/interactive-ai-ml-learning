import React, { useRef, useEffect } from 'react';
import * as stats from '../../utils/probabilityStatistics';

export default function InteractiveDescriptiveStatsVisualization({ dataSet }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !dataSet || dataSet.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 600;
    const displayHeight = rect.height > 0 ? rect.height : 300;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const padding = { top: 40, right: 20, bottom: 60, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const statsData = stats.calculateDescriptiveStats(dataSet);
    const min = statsData.min;
    const max = statsData.max;
    const range = statsData.range || 1;
    const mean = statsData.mean;
    const stdDev = statsData.standardDeviation;

    // Create histogram bins
    const numBins = Math.min(20, Math.ceil(Math.sqrt(dataSet.length)));
    const binWidth = range / numBins;
    const bins = Array(numBins).fill(0);
    
    dataSet.forEach(value => {
      let binIndex = Math.floor((value - min) / binWidth);
      if (binIndex >= numBins) binIndex = numBins - 1;
      if (binIndex < 0) binIndex = 0;
      bins[binIndex]++;
    });

    const maxCount = Math.max(...bins);

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * (height - padding.top - padding.bottom);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw histogram bars
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const barWidth = plotWidth / numBins;

    ctx.fillStyle = '#3b82f6';
    bins.forEach((count, i) => {
      const barHeight = (count / maxCount) * plotHeight;
      const x = padding.left + i * barWidth;
      const y = height - padding.bottom - barHeight;
      
      ctx.fillRect(x, y, barWidth - 2, barHeight);
    });

    // Draw mean line
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    const meanX = padding.left + ((mean - min) / range) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(meanX, padding.top);
    ctx.lineTo(meanX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw mean label
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Mean', meanX, padding.top - 10);

    // Draw standard deviation bands
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const stdDev1X = padding.left + ((mean + stdDev - min) / range) * plotWidth;
    const stdDev2X = padding.left + ((mean - stdDev - min) / range) * plotWidth;
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
    
    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const value = min + (i / 5) * range;
      const x = padding.left + (i / 5) * plotWidth;
      ctx.fillText(value.toFixed(1), x, height - padding.bottom + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const count = Math.round((i / 5) * maxCount);
      const y = padding.top + (i / 5) * plotHeight;
      ctx.fillText(count.toString(), padding.left - 10, y + 4);
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Data Distribution', width / 2, 20);

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    
    ctx.strokeStyle = '#ef4444';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 20);
    ctx.lineTo(width - padding.right - 50, padding.top + 20);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ef4444';
    ctx.fillText('Mean', width - padding.right - 45, padding.top + 25);

    ctx.strokeStyle = '#f59e0b';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(width - padding.right - 100, padding.top + 40);
    ctx.lineTo(width - padding.right - 50, padding.top + 40);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('±1 Std Dev', width - padding.right - 45, padding.top + 45);
  }, [dataSet]);

  if (!dataSet || dataSet.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
        Please provide data to visualize
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> This histogram shows your data distribution with mean and standard deviation bands!
        </p>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '300px' }}
      />
    </div>
  );
}

