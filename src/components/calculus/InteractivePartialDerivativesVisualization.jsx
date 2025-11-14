import React, { useRef, useEffect, useState } from 'react';

export default function InteractivePartialDerivativesVisualization() {
  const canvasRef = useRef(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [viewAngle, setViewAngle] = useState(45);

  // Sample function: f(x, y) = x² + y² (paraboloid)
  const f = (x, y) => x * x + y * y;
  const partialX = (x, y) => 2 * x;
  const partialY = (x, y) => 2 * y;

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
    const centerX = width / 2;
    const centerY = height / 2;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const xMin = -3;
    const xMax = 3;
    const yMin = -3;
    const yMax = 3;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    // 3D projection
    const angle = (viewAngle * Math.PI) / 180;
    const cosAngle = Math.cos(angle);
    const sinAngle = Math.sin(angle);

    const project3D = (x, y, z) => {
      const x2D = x * cosAngle - y * sinAngle;
      const y2D = x * sinAngle + y * cosAngle + z * 0.3;
      return {
        x: centerX + (x2D / 6) * plotWidth,
        y: centerY - (y2D / 6) * plotHeight
      };
    };

    // Draw contour lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let level = 0; level <= 20; level += 2) {
      ctx.beginPath();
      let firstPoint = true;
      for (let angle = 0; angle <= 2 * Math.PI; angle += 0.1) {
        const r = Math.sqrt(level);
        const x = r * Math.cos(angle);
        const y = r * Math.sin(angle);
        if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
          const z = f(x, y);
          const proj = project3D(x, y, z);
          if (firstPoint) {
            ctx.moveTo(proj.x, proj.y);
            firstPoint = false;
          } else {
            ctx.lineTo(proj.x, proj.y);
          }
        }
      }
      ctx.stroke();
    }

    // Draw surface (simplified as wireframe)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1;
    const step = 0.3;
    for (let x = xMin; x <= xMax; x += step) {
      ctx.beginPath();
      let firstPoint = true;
      for (let y = yMin; y <= yMax; y += step * 0.5) {
        const z = f(x, y);
        const proj = project3D(x, y, z);
        if (firstPoint) {
          ctx.moveTo(proj.x, proj.y);
          firstPoint = false;
        } else {
          ctx.lineTo(proj.x, proj.y);
        }
      }
      ctx.stroke();
    }

    for (let y = yMin; y <= yMax; y += step) {
      ctx.beginPath();
      let firstPoint = true;
      for (let x = xMin; x <= xMax; x += step * 0.5) {
        const z = f(x, y);
        const proj = project3D(x, y, z);
        if (firstPoint) {
          ctx.moveTo(proj.x, proj.y);
          firstPoint = false;
        } else {
          ctx.lineTo(proj.x, proj.y);
        }
      }
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    const origin = project3D(0, 0, 0);
    const xAxis = project3D(3, 0, 0);
    const yAxis = project3D(0, 3, 0);
    const zAxis = project3D(0, 0, 10);

    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(xAxis.x, xAxis.y);
    ctx.stroke();
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.fillText('x', xAxis.x + 5, xAxis.y);

    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(yAxis.x, yAxis.y);
    ctx.stroke();
    ctx.fillText('y', yAxis.x + 5, yAxis.y);

    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(zAxis.x, zAxis.y);
    ctx.stroke();
    ctx.fillText('z', zAxis.x + 5, zAxis.y);

    // Draw hover point and partial derivatives
    if (hoveredPoint) {
      const { x, y } = hoveredPoint;
      const z = f(x, y);
      const proj = project3D(x, y, z);
      const px = partialX(x, y);
      const py = partialY(x, y);

      // Draw point
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Draw gradient vector
      const scale = 0.5;
      const gradProj = project3D(x + px * scale, y + py * scale, z + 0.5);
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(proj.x, proj.y);
      ctx.lineTo(gradProj.x, gradProj.y);
      ctx.stroke();

      // Draw arrowhead
      const angle = Math.atan2(gradProj.y - proj.y, gradProj.x - proj.x);
      ctx.save();
      ctx.translate(gradProj.x, gradProj.y);
      ctx.rotate(angle);
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-10, -5);
      ctx.lineTo(-10, 5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Draw labels
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`(x, y) = (${x.toFixed(2)}, ${y.toFixed(2)})`, proj.x + 10, proj.y - 20);
      ctx.fillText(`f(x, y) = ${z.toFixed(2)}`, proj.x + 10, proj.y - 5);
      ctx.fillStyle = '#10b981';
      ctx.fillText(`∂f/∂x = ${px.toFixed(2)}`, proj.x + 10, proj.y + 10);
      ctx.fillText(`∂f/∂y = ${py.toFixed(2)}`, proj.x + 10, proj.y + 25);
    }
  }, [hoveredPoint, viewAngle]);

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert screen coordinates to world coordinates (simplified)
    const worldX = ((x - centerX) / (rect.width / 2)) * 3;
    const worldY = ((centerY - y) / (rect.height / 2)) * 3;

    if (Math.abs(worldX) <= 3 && Math.abs(worldY) <= 3) {
      setHoveredPoint({ x: worldX, y: worldY });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="font-mono text-sm space-y-2">
          <div>f(x, y) = x² + y²</div>
          <div>∂f/∂x = 2x</div>
          <div>∂f/∂y = 2y</div>
        </div>
      </div>
      <div className="flex items-center gap-4 mb-2">
        <label className="text-sm font-semibold text-gray-700">View Angle:</label>
        <input
          type="range"
          min="0"
          max="90"
          value={viewAngle}
          onChange={(e) => setViewAngle(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-sm text-gray-600 w-12">{viewAngle}°</span>
      </div>
      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Hover over the surface to see partial derivatives! Adjust the view angle to see different perspectives.
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

