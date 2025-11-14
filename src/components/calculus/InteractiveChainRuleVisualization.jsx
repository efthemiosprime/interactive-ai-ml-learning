import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCw, ArrowRight, ArrowDown } from 'lucide-react';

export default function InteractiveChainRuleVisualization() {
  const canvasRef = useRef(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);

  // Example: f(g(x)) where g(x) = x² and f(u) = sin(u)
  // So f(g(x)) = sin(x²)
  // Chain rule: d/dx[sin(x²)] = cos(x²) × 2x
  const g = (x) => x * x;
  const f = (u) => Math.sin(u);
  const composite = (x) => f(g(x));
  const gPrime = (x) => 2 * x;
  const fPrime = (u) => Math.cos(u);
  const compositePrime = (x) => fPrime(g(x)) * gPrime(x);

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
    const xRange = xMax - xMin;

    // Calculate y range
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let x = xMin; x <= xMax; x += 0.1) {
      const y = composite(x);
      yMin = Math.min(yMin, y);
      yMax = Math.max(yMax, y);
    }
    const yRange = yMax - yMin;
    yMin -= yRange * 0.1;
    yMax += yRange * 0.1;
    const adjustedYRange = yMax - yMin;

    const toScreenX = (x) => padding + ((x - xMin) / xRange) * plotWidth;
    const toScreenY = (y) => padding + plotHeight - ((y - yMin) / adjustedYRange) * plotHeight;

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

    // Draw composite function
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    let firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.05) {
      const y = composite(x);
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

    // Draw derivative
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    firstPoint = true;
    for (let x = xMin; x <= xMax; x += 0.05) {
      const y = compositePrime(x);
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

    // Animation: show chain rule step by step
    if (isAnimating) {
      const x = -2 + (animationStep / 30) * 4; // Animate from -2 to 2
      if (x >= xMin && x <= xMax) {
        const u = g(x);
        const y = composite(x);
        const screenX = toScreenX(x);
        const screenY = toScreenY(y);

        // Draw point on composite function
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(screenX, screenY, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Draw tangent line
        const slope = compositePrime(x);
        const lineLength = 100;
        const dx = lineLength / Math.sqrt(1 + slope * slope);
        const dy = slope * dx;
        const scale = plotWidth / xRange;
        const scaledDx = dx * scale;
        const scaledDy = -dy * scale;

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(screenX - scaledDx, screenY - scaledDy);
        ctx.lineTo(screenX + scaledDx, screenY + scaledDy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw labels showing chain rule
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`x = ${x.toFixed(2)}`, screenX + 10, screenY - 30);
        ctx.fillText(`g(x) = ${u.toFixed(2)}`, screenX + 10, screenY - 15);
        ctx.fillText(`f(g(x)) = ${y.toFixed(2)}`, screenX + 10, screenY);
        ctx.fillText(`g'(x) = ${gPrime(x).toFixed(2)}`, screenX + 10, screenY + 15);
        ctx.fillText(`f'(g(x)) = ${fPrime(u).toFixed(2)}`, screenX + 10, screenY + 30);
        ctx.fillStyle = '#10b981';
        ctx.fillText(`d/dx = ${compositePrime(x).toFixed(2)}`, screenX + 10, screenY + 45);
      }
    }

    // Draw legend
    ctx.fillStyle = '#3b82f6';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('f(g(x)) = sin(x²)', width - padding - 150, padding + 20);

    ctx.strokeStyle = '#10b981';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(width - padding - 150, padding + 35);
    ctx.lineTo(width - padding - 50, padding + 35);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#10b981';
    ctx.fillText("d/dx[f(g(x))]", width - padding - 150, padding + 50);
  }, [isAnimating, animationStep]);

  useEffect(() => {
    if (isAnimating) {
      let animationFrameId;
      let lastTime = 0;

      const animate = (currentTime) => {
        if (currentTime - lastTime >= 50) {
          setAnimationStep((prev) => {
            const next = prev + 1;
            if (next >= 60) {
              setIsAnimating(false);
              return 0;
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
  }, [isAnimating]);

  const reset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="font-mono text-sm space-y-2">
          <div>g(x) = x²</div>
          <div>f(u) = sin(u)</div>
          <div>f(g(x)) = sin(x²)</div>
          <div className="mt-2 pt-2 border-t border-gray-300">
            <div className="text-purple-700 font-bold">Chain Rule:</div>
            <div>d/dx[f(g(x))] = f'(g(x)) × g'(x)</div>
            <div>= cos(x²) × 2x</div>
          </div>
        </div>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2"
        >
          {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isAnimating ? 'Pause' : 'Animate Chain Rule'}
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
          💡 <strong>Interactive:</strong> Click "Animate Chain Rule" to see how the derivative is computed step-by-step!
        </p>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Visual Flow Diagram */}
      <div className="bg-indigo-50 rounded-lg p-4 border-2 border-indigo-200 mt-4">
        <h4 className="font-bold text-indigo-900 mb-3 text-center">Chain Rule Flow</h4>
        <div className="flex items-center justify-center gap-4">
          <div className="text-center">
            <div className="bg-blue-100 rounded-lg p-3 mb-2">
              <div className="text-xs font-semibold text-blue-900 mb-1">Input</div>
              <div className="text-xs font-mono">x</div>
            </div>
            <div className="text-xs text-gray-600">g'(x) = 2x</div>
          </div>
          <ArrowRight className="w-6 h-6 text-blue-500" />
          <div className="text-center">
            <div className="bg-purple-100 rounded-lg p-3 mb-2">
              <div className="text-xs font-semibold text-purple-900 mb-1">Intermediate</div>
              <div className="text-xs font-mono">u = g(x)</div>
            </div>
            <div className="text-xs text-gray-600">f'(u) = cos(u)</div>
          </div>
          <ArrowRight className="w-6 h-6 text-purple-500" />
          <div className="text-center">
            <div className="bg-green-100 rounded-lg p-3 mb-2">
              <div className="text-xs font-semibold text-green-900 mb-1">Output</div>
              <div className="text-xs font-mono">f(g(x))</div>
            </div>
            <div className="text-xs text-gray-600">d/dx = f'(g(x)) × g'(x)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

