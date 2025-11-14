import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCw } from 'lucide-react';

export default function InteractiveGradientVisualization() {
  const canvasRef = useRef(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [learningRate, setLearningRate] = useState(0.1);

  // Loss function: L(x, y) = (x-1)² + (y-1)² (minimum at (1, 1))
  const lossFunction = (x, y) => Math.pow(x - 1, 2) + Math.pow(y - 1, 2);
  const gradientX = (x, y) => 2 * (x - 1);
  const gradientY = (x, y) => 2 * (y - 1);

  // Gradient descent path
  const [path, setPath] = useState([{ x: -2, y: -2 }]);

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

    const xMin = -3;
    const xMax = 3;
    const yMin = -3;
    const yMax = 3;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toScreenX = (x) => padding + ((x - xMin) / xRange) * plotWidth;
    const toScreenY = (y) => padding + plotHeight - ((y - yMin) / yRange) * plotHeight;

    // Draw contour lines (loss surface)
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let level = 0.5; level <= 10; level += 0.5) {
      ctx.beginPath();
      let firstPoint = true;
      for (let angle = 0; angle <= 2 * Math.PI; angle += 0.05) {
        const r = Math.sqrt(level);
        const x = 1 + r * Math.cos(angle);
        const y = 1 + r * Math.sin(angle);
        if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
          const screenX = toScreenX(x);
          const screenY = toScreenY(y);
          if (firstPoint) {
            ctx.moveTo(screenX, screenY);
            firstPoint = false;
          } else {
            ctx.lineTo(screenX, screenY);
          }
        }
      }
      ctx.stroke();
    }

    // Draw gradient vector field
    const gridStep = 0.5;
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1;
    for (let x = xMin; x <= xMax; x += gridStep) {
      for (let y = yMin; y <= yMax; y += gridStep) {
        const gx = gradientX(x, y);
        const gy = gradientY(x, y);
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        if (magnitude > 0.1) {
          const scale = 0.15 / magnitude;
          const screenX = toScreenX(x);
          const screenY = toScreenY(y);
          const endX = screenX + gx * scale * plotWidth / xRange;
          const endY = screenY - gy * scale * plotHeight / yRange;

          ctx.beginPath();
          ctx.moveTo(screenX, screenY);
          ctx.lineTo(endX, endY);
          ctx.stroke();

          // Arrowhead
          const angle = Math.atan2(screenY - endY, endX - screenX);
          ctx.save();
          ctx.translate(endX, endY);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(-5, -3);
          ctx.lineTo(-5, 3);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }
      }
    }

    // Draw gradient descent path
    if (path.length > 1) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(toScreenX(path[0].x), toScreenY(path[0].y));
      for (let i = 1; i < path.length; i++) {
        ctx.lineTo(toScreenX(path[i].x), toScreenY(path[i].y));
      }
      ctx.stroke();

      // Draw points
      path.forEach((point, i) => {
        ctx.fillStyle = i === 0 ? '#10b981' : i === path.length - 1 ? '#ef4444' : '#f59e0b';
        ctx.beginPath();
        ctx.arc(toScreenX(point.x), toScreenY(point.y), 5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Draw current position (animated)
    if (isAnimating && path.length > 0) {
      const currentPoint = path[Math.min(animationStep, path.length - 1)];
      const screenX = toScreenX(currentPoint.x);
      const screenY = toScreenY(currentPoint.y);
      const gx = gradientX(currentPoint.x, currentPoint.y);
      const gy = gradientY(currentPoint.x, currentPoint.y);

      // Draw current point
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(screenX, screenY, 8, 0, 2 * Math.PI);
      ctx.fill();

      // Draw gradient vector
      const scale = 0.3;
      const endX = screenX + gx * scale * plotWidth / xRange;
      const endY = screenY - gy * scale * plotHeight / yRange;

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(screenX, screenY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Arrowhead
      const angle = Math.atan2(screenY - endY, endX - screenX);
      ctx.save();
      ctx.translate(endX, endY);
      ctx.rotate(angle);
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-8, -5);
      ctx.lineTo(-8, 5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Label
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Step ${animationStep}`, screenX + 10, screenY - 10);
      ctx.fillText(`Loss: ${lossFunction(currentPoint.x, currentPoint.y).toFixed(3)}`, screenX + 10, screenY + 5);
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    const zeroY = toScreenY(0);
    const zeroX = toScreenX(0);

    ctx.beginPath();
    ctx.moveTo(padding, zeroY);
    ctx.lineTo(width - padding, zeroY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(zeroX, padding);
    ctx.lineTo(zeroX, height - padding);
    ctx.stroke();

    // Draw minimum point
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(toScreenX(1), toScreenY(1), 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Minimum (1, 1)', toScreenX(1), toScreenY(1) - 15);
  }, [path, isAnimating, animationStep, learningRate]);

  useEffect(() => {
    if (isAnimating) {
      let animationFrameId;
      let lastTime = 0;

      const animate = (currentTime) => {
        if (currentTime - lastTime >= 100) {
          setAnimationStep((prev) => {
            const next = prev + 1;
            if (next >= path.length) {
              setIsAnimating(false);
              return prev;
            }
            return next;
          });
          lastTime = currentTime;
        }
        animationFrameId = requestAnimationFrame(animate);
      };

      animationFrameId = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationFrameId);
    }
  }, [isAnimating, path.length]);

  const startGradientDescent = () => {
    const newPath = [{ x: -2, y: -2 }];
    let currentX = -2;
    let currentY = -2;
    const maxSteps = 20;

    for (let i = 0; i < maxSteps; i++) {
      const gx = gradientX(currentX, currentY);
      const gy = gradientY(currentX, currentY);
      const magnitude = Math.sqrt(gx * gx + gy * gy);

      if (magnitude < 0.01) break;

      currentX = currentX - learningRate * gx;
      currentY = currentY - learningRate * gy;
      newPath.push({ x: currentX, y: currentY });
    }

    setPath(newPath);
    setAnimationStep(0);
  };

  const reset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
    setPath([{ x: -2, y: -2 }]);
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="font-mono text-sm space-y-2">
          <div>L(x, y) = (x-1)² + (y-1)²</div>
          <div>∇L = [2(x-1), 2(y-1)]</div>
          <div>Update: (x, y) = (x, y) - α∇L</div>
        </div>
      </div>

      <div className="flex items-center gap-4 mb-2">
        <label className="text-sm font-semibold text-gray-700">Learning Rate (α):</label>
        <input
          type="range"
          min="0.01"
          max="0.3"
          step="0.01"
          value={learningRate}
          onChange={(e) => setLearningRate(Number(e.target.value))}
          className="flex-1"
          disabled={isAnimating}
        />
        <span className="text-sm text-gray-600 w-16">{learningRate.toFixed(2)}</span>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={startGradientDescent}
          disabled={isAnimating}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          Start Gradient Descent
        </button>
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          disabled={path.length <= 1}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isAnimating ? 'Pause' : 'Animate'}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Click "Start Gradient Descent" to compute the path, then "Animate" to watch the optimization process!
        </p>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />
    </div>
  );
}

