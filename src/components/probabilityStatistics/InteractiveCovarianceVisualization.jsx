import React, { useRef, useEffect, useState } from 'react';
import * as stats from '../../utils/probabilityStatistics';
import * as math from '../../utils/math';

export default function InteractiveCovarianceVisualization() {
  const canvasRef = useRef(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [dataPoints, setDataPoints] = useState(() => {
    // Generate sample data with correlation
    const points = [];
    for (let i = 0; i < 30; i++) {
      const x = Math.random() * 10 - 5;
      const y = 0.7 * x + (Math.random() - 0.5) * 3; // Positive correlation with noise
      points.push({ x, y });
    }
    return points;
  });

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

    // Calculate statistics
    const xValues = dataPoints.map(p => p.x);
    const yValues = dataPoints.map(p => p.y);
    const xMean = math.mean(xValues);
    const yMean = math.mean(yValues);
    const covariance = math.covariance(xValues, yValues);
    const xStdDev = math.standardDeviation(xValues);
    const yStdDev = math.standardDeviation(yValues);
    const correlation = xStdDev > 0 && yStdDev > 0 
      ? covariance / (xStdDev * yStdDev) 
      : 0;

    // Find ranges
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    // Convert to screen coordinates
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
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * (height - padding.top - padding.bottom);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding.left, toScreenY(0));
    ctx.lineTo(width - padding.right, toScreenY(0));
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(toScreenX(0), padding.top);
    ctx.lineTo(toScreenX(0), height - padding.bottom);
    ctx.stroke();

    // Draw mean lines
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    const meanX = toScreenX(xMean);
    const meanY = toScreenY(yMean);
    ctx.beginPath();
    ctx.moveTo(meanX, padding.top);
    ctx.lineTo(meanX, height - padding.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(padding.left, meanY);
    ctx.lineTo(width - padding.right, meanY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw data points
    dataPoints.forEach((point, i) => {
      const screenX = toScreenX(point.x);
      const screenY = toScreenY(point.y);
      const isHovered = hoveredPoint === i;

      ctx.fillStyle = isHovered ? '#ef4444' : '#3b82f6';
      ctx.beginPath();
      ctx.arc(screenX, screenY, isHovered ? 6 : 4, 0, 2 * Math.PI);
      ctx.fill();

      // Draw line to mean if hovered
      if (isHovered) {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(screenX, screenY);
        ctx.lineTo(meanX, meanY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i <= 5; i++) {
      const value = xMin + (i / 5) * xRange;
      const x = padding.left + (i / 5) * (width - padding.left - padding.right);
      ctx.fillText(value.toFixed(1), x, height - padding.bottom + 20);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const value = yMin + (i / 5) * yRange;
      const y = padding.top + (5 - i) / 5 * (height - padding.top - padding.bottom);
      ctx.fillText(value.toFixed(1), padding.left - 10, y + 4);
    }

    // Draw title and stats
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Covariance & Correlation Scatter Plot', width / 2, 20);

    // Draw statistics box
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(width - padding.right - 180, padding.top, 170, 100);
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(width - padding.right - 180, padding.top, 170, 100);

    ctx.textAlign = 'left';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Statistics', width - padding.right - 175, padding.top + 18);

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText(`Covariance: ${covariance.toFixed(3)}`, width - padding.right - 175, padding.top + 35);
    ctx.fillText(`Correlation: ${correlation.toFixed(3)}`, width - padding.right - 175, padding.top + 50);
    ctx.fillText(`X Mean: ${xMean.toFixed(2)}`, width - padding.right - 175, padding.top + 65);
    ctx.fillText(`Y Mean: ${yMean.toFixed(2)}`, width - padding.right - 175, padding.top + 80);

    // Draw hover info
    if (hoveredPoint !== null) {
      const point = dataPoints[hoveredPoint];
      const screenX = toScreenX(point.x);
      const screenY = toScreenY(point.y);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(screenX + 10, screenY - 50, 120, 40);
      
      ctx.fillStyle = '#ffffff';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`X: ${point.x.toFixed(2)}`, screenX + 15, screenY - 35);
      ctx.fillText(`Y: ${point.y.toFixed(2)}`, screenX + 15, screenY - 20);
    }
  }, [dataPoints, hoveredPoint]);

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Find closest point
    let closestIndex = null;
    let minDistance = Infinity;

    const xValues = dataPoints.map(p => p.x);
    const yValues = dataPoints.map(p => p.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const toWorldX = (screenX) => xMin + ((screenX - padding.left) / (rect.width - padding.left - padding.right)) * xRange;
    const toWorldY = (screenY) => yMin + ((rect.height - padding.top - padding.bottom - (screenY - padding.top)) / (rect.height - padding.top - padding.bottom)) * yRange;

    const worldX = toWorldX(x);
    const worldY = toWorldY(y);

    dataPoints.forEach((point, i) => {
      const dx = point.x - worldX;
      const dy = point.y - worldY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < minDistance && distance < 0.5) {
        minDistance = distance;
        closestIndex = i;
      }
    });

    setHoveredPoint(closestIndex);
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  const generateNewData = () => {
    const points = [];
    for (let i = 0; i < 30; i++) {
      const x = Math.random() * 10 - 5;
      const y = 0.7 * x + (Math.random() - 0.5) * 3;
      points.push({ x, y });
    }
    setDataPoints(points);
    setHoveredPoint(null);
  };

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Hover over points to see their values and relationship to the mean!
        </p>
      </div>
      <div className="flex gap-2 mb-2">
        <button
          onClick={generateNewData}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Generate New Data
        </button>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg cursor-crosshair"
        style={{ height: '400px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}

