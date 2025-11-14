import React, { useRef, useEffect, useState } from 'react';

export default function InteractiveDerivativeVisualization({ functionType }) {
  const canvasRef = useRef(null);
  const [hoveredX, setHoveredX] = useState(null);

  const getFunction = (type) => {
    switch (type) {
      case 'quadratic':
        return (x) => x * x;
      case 'cubic':
        return (x) => x * x * x;
      case 'sine':
        return (x) => Math.sin(x);
      case 'exponential':
        return (x) => Math.exp(x);
      default:
        return (x) => x * x;
    }
  };

  const getDerivative = (type) => {
    switch (type) {
      case 'quadratic':
        return (x) => 2 * x;
      case 'cubic':
        return (x) => 3 * x * x;
      case 'sine':
        return (x) => Math.cos(x);
      case 'exponential':
        return (x) => Math.exp(x);
      default:
        return (x) => 2 * x;
    }
  };

  const f = getFunction(functionType);
  const fPrime = getDerivative(functionType);

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
    const padding = 60;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Domain and range
    const xMin = -5;
    const xMax = 5;
    const xRange = xMax - xMin;
    
    // Calculate y range based on function values
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let x = xMin; x <= xMax; x += 0.1) {
      const y = f(x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
    // Add padding
    const yRange = yMax - yMin;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;
    const adjustedYRange = yMax - yMin;

    // Convert coordinates
    const toScreenX = (x) => padding + ((x - xMin) / xRange) * plotWidth;
    const toScreenY = (y) => padding + plotHeight - ((y - yMin) / adjustedYRange) * plotHeight;
    const toWorldX = (screenX) => xMin + ((screenX - padding) / plotWidth) * xRange;
    const toWorldY = (screenY) => yMin + ((plotHeight - (screenY - padding)) / plotHeight) * adjustedYRange;

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = xMin + (i / 10) * xRange;
      const screenX = toScreenX(x);
      ctx.beginPath();
      ctx.moveTo(screenX, padding);
      ctx.lineTo(screenX, height - padding);
      ctx.stroke();
    }
    for (let i = 0; i <= 10; i++) {
      const y = yMin + (i / 10) * adjustedYRange;
      const screenY = toScreenY(y);
      ctx.beginPath();
      ctx.moveTo(padding, screenY);
      ctx.lineTo(width - padding, screenY);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    const zeroY = toScreenY(0);
    const zeroX = toScreenX(0);
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(width - padding, zeroY);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(zeroX, padding);
    ctx.lineTo(zeroX, height - padding);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    for (let i = -5; i <= 5; i += 1) {
      if (i !== 0) {
        const screenX = toScreenX(i);
        ctx.fillText(i.toString(), screenX, zeroY + 20);
      }
    }
    ctx.textAlign = 'right';
    for (let i = Math.ceil(yMin); i <= Math.floor(yMax); i += 1) {
      if (i !== 0) {
        const screenY = toScreenY(i);
        ctx.fillText(i.toString(), zeroX - 10, screenY + 4);
      }
    }

    // Draw function curve
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    let firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.05) {
      const y = f(x);
      const screenX = toScreenX(x);
      const screenY = toScreenY(y);
      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();

    // Draw derivative curve
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.05) {
      const y = fPrime(x);
      const screenX = toScreenX(x);
      const screenY = toScreenY(y);
      if (firstPoint) {
        ctx.moveTo(screenX, screenY);
        firstPoint = false;
      } else {
        ctx.lineTo(screenX, screenY);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw hover point and tangent line
    if (hoveredX !== null) {
      const x = toWorldX(hoveredX);
      const y = f(x);
      const slope = fPrime(x);
      const screenX = toScreenX(x);
      const screenY = toScreenY(y);

      // Draw point
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(screenX, screenY, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw tangent line
      const lineLength = 100;
      const dx = lineLength / Math.sqrt(1 + slope * slope);
      const dy = slope * dx;
      const scale = plotWidth / xRange;
      const scaledDx = dx * scale;
      const scaledDy = -dy * scale; // Negative because y increases downward

      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(screenX - scaledDx, screenY - scaledDy);
      ctx.lineTo(screenX + scaledDx, screenY + scaledDy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw label
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`x = ${x.toFixed(2)}`, screenX + 10, screenY - 10);
      ctx.fillText(`f(x) = ${y.toFixed(2)}`, screenX + 10, screenY + 5);
      ctx.fillText(`f'(x) = ${slope.toFixed(2)}`, screenX + 10, screenY + 20);
    }

    // Draw legend
    ctx.fillStyle = '#3b82f6';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('f(x)', width - padding - 100, padding + 20);
    
    ctx.strokeStyle = '#10b981';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding - 100, padding + 35);
    ctx.lineTo(width - padding - 50, padding + 35);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#10b981';
    ctx.fillText("f'(x)", width - padding - 100, padding + 50);
  }, [functionType, hoveredX, f, fPrime]);

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const padding = 60;
    const x = e.clientX - rect.left;
    
    if (x >= padding && x <= rect.width - padding) {
      setHoveredX(x);
    } else {
      setHoveredX(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredX(null);
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="font-mono text-lg mb-2">
          f'(x) = {
            functionType === 'quadratic' && '2x'
          }
          {functionType === 'cubic' && '3x²'}
          {functionType === 'sine' && 'cos(x)'}
          {functionType === 'exponential' && 'eˣ'}
        </div>
      </div>
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Hover over the graph to see the tangent line and derivative value at any point!
        </p>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg cursor-crosshair"
        style={{ height: '500px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}

