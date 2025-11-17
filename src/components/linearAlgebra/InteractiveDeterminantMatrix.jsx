import React, { useState, useRef, useEffect } from 'react';
import { Info } from 'lucide-react';
import * as math from '../../utils/math';

export default function InteractiveDeterminantMatrix({ matrix }) {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [selectedRow, setSelectedRow] = useState(null);
  const [selectedCol, setSelectedCol] = useState(null);
  const canvasRef = useRef(null);

  const isSquare = matrix.length > 0 && matrix.length === matrix[0].length;

  // Canvas visualization showing geometric interpretation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isSquare || matrix.length > 3) return; // Only visualize 2x2 and 3x3

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
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = 40; // Pixels per unit

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (matrix.length === 2) {
      // 2D visualization: parallelogram formed by column vectors
      const v1 = [matrix[0][0], matrix[1][0]]; // First column
      const v2 = [matrix[0][1], matrix[1][1]]; // Second column
      const det = math.determinant(matrix);

      // Draw axes
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX - width/3, centerY);
      ctx.lineTo(centerX + width/3, centerY);
      ctx.moveTo(centerX, centerY - height/3);
      ctx.lineTo(centerX, centerY + height/3);
      ctx.stroke();

      // Draw origin vectors
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + v1[0] * scale, centerY - v1[1] * scale);
      ctx.stroke();
      ctx.fillStyle = '#3b82f6';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('v₁', centerX + v1[0] * scale + 5, centerY - v1[1] * scale - 5);

      ctx.strokeStyle = '#10b981';
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + v2[0] * scale, centerY - v2[1] * scale);
      ctx.stroke();
      ctx.fillStyle = '#10b981';
      ctx.fillText('v₂', centerX + v2[0] * scale + 5, centerY - v2[1] * scale - 5);

      // Draw parallelogram
      const p1 = [centerX, centerY];
      const p2 = [centerX + v1[0] * scale, centerY - v1[1] * scale];
      const p3 = [centerX + v1[0] * scale + v2[0] * scale, centerY - v1[1] * scale - v2[1] * scale];
      const p4 = [centerX + v2[0] * scale, centerY - v2[1] * scale];

      ctx.fillStyle = det === 0 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(59, 130, 246, 0.2)';
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]);
      ctx.lineTo(p4[0], p4[1]);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = det === 0 ? '#ef4444' : '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]);
      ctx.lineTo(p4[0], p4[1]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw area label
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        `Area = |det| = ${Math.abs(det).toFixed(2)}`,
        centerX,
        centerY + height/3 + 20
      );
      if (det === 0) {
        ctx.fillStyle = '#ef4444';
        ctx.fillText('Vectors are collinear (linearly dependent)', centerX, centerY + height/3 + 40);
      }
    } else if (matrix.length === 3) {
      // 3D visualization: parallelepiped (simplified 2D projection)
      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('3D Parallelepiped (2D projection)', centerX, 30);
      ctx.fillText(`Volume = |det| = ${Math.abs(math.determinant(matrix)).toFixed(2)}`, centerX, 50);
      
      if (math.determinant(matrix) === 0) {
        ctx.fillStyle = '#ef4444';
        ctx.fillText('Vectors are coplanar (linearly dependent)', centerX, 70);
      }
    }
  }, [matrix, isSquare]);

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
      <h4 className="font-bold text-indigo-900 mb-4 text-center">Interactive Matrix</h4>
      <p className="text-xs text-gray-600 mb-4 text-center">
        Hover over cells to see coordinates. Click rows/columns to highlight them.
      </p>
      
      <div className="bg-white rounded-lg p-4 border-2 border-indigo-300">
        <div className="font-mono text-sm flex justify-center">
          <div className="border-2 border-indigo-400 rounded p-3">
            {/* Column headers */}
            <div className="flex gap-2 justify-center mb-1">
              <div className="w-16"></div>
              {matrix[0]?.map((_, j) => (
                <div
                  key={j}
                  onClick={() => setSelectedCol(selectedCol === j ? null : j)}
                  className={`w-16 text-center px-2 py-1 rounded text-xs font-semibold cursor-pointer transition-all ${
                    selectedCol === j
                      ? 'bg-purple-600 text-white ring-2 ring-purple-800'
                      : 'bg-purple-100 text-purple-900 hover:bg-purple-200'
                  }`}
                >
                  Col {j + 1}
                </div>
              ))}
            </div>
            
            {matrix.map((row, i) => (
              <div key={i} className="flex gap-2 justify-center mb-1">
                {/* Row header */}
                <div
                  onClick={() => setSelectedRow(selectedRow === i ? null : i)}
                  className={`w-16 text-center px-2 py-1 rounded text-xs font-semibold cursor-pointer transition-all flex items-center justify-center ${
                    selectedRow === i
                      ? 'bg-indigo-600 text-white ring-2 ring-indigo-800'
                      : 'bg-indigo-100 text-indigo-900 hover:bg-indigo-200'
                  }`}
                >
                  Row {i + 1}
                </div>
                
                {/* Matrix cells */}
                {row.map((val, j) => {
                  const cellKey = `${i}-${j}`;
                  const isHovered = hoveredCell === cellKey;
                  const isInSelectedRow = selectedRow === i;
                  const isInSelectedCol = selectedCol === j;
                  
                  return (
                    <span
                      key={j}
                      onMouseEnter={() => setHoveredCell(cellKey)}
                      onMouseLeave={() => setHoveredCell(null)}
                      className={`w-16 text-center px-2 py-1 rounded font-semibold transition-all cursor-pointer ${
                        isHovered
                          ? 'bg-yellow-400 text-gray-900 ring-4 ring-yellow-600 scale-110 z-10 relative'
                          : isInSelectedRow && isInSelectedCol
                          ? 'bg-purple-300 text-purple-900 ring-2 ring-purple-500'
                          : isInSelectedRow
                          ? 'bg-indigo-200 text-indigo-900'
                          : isInSelectedCol
                          ? 'bg-purple-200 text-purple-900'
                          : 'bg-indigo-100 text-indigo-900 hover:bg-indigo-200'
                      }`}
                      title={`A[${i}][${j}] = ${val.toFixed(2)}`}
                    >
                      {val.toFixed(1)}
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cell Info */}
      {hoveredCell && (
        <div className="mt-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-900">
              <strong>Cell:</strong> A[{hoveredCell.split('-')[0]}][{hoveredCell.split('-')[1]}] = {matrix[parseInt(hoveredCell.split('-')[0])][parseInt(hoveredCell.split('-')[1])].toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Selection Info */}
      {(selectedRow !== null || selectedCol !== null) && (
        <div className="mt-4 bg-blue-50 border-2 border-blue-300 rounded-lg p-3">
          <div className="text-sm text-blue-900">
            {selectedRow !== null && (
              <div>
                <strong>Selected Row {selectedRow + 1}:</strong> [
                {matrix[selectedRow].map((val, idx) => (
                  <span key={idx} className="font-mono mx-1">{val.toFixed(1)}</span>
                ))}
                ]
              </div>
            )}
            {selectedCol !== null && (
              <div className="mt-2">
                <strong>Selected Column {selectedCol + 1}:</strong> [
                {matrix.map((row, idx) => (
                  <span key={idx} className="font-mono mx-1">{row[selectedCol].toFixed(1)}</span>
                ))}
                ]
              </div>
            )}
          </div>
        </div>
      )}

      {/* Canvas Geometric Visualization */}
      {isSquare && (matrix.length === 2 || matrix.length === 3) && (
        <div className="mt-6 bg-white rounded-lg p-4 border-2 border-indigo-300">
          <h5 className="font-bold text-indigo-900 mb-3 text-center">Geometric Interpretation</h5>
          <p className="text-xs text-gray-600 mb-3 text-center">
            {matrix.length === 2 
              ? 'The determinant represents the area of the parallelogram formed by the column vectors.'
              : 'The determinant represents the volume of the parallelepiped formed by the column vectors.'}
          </p>
          <canvas
            ref={canvasRef}
            className="w-full border border-gray-300 rounded-lg"
            style={{ height: '400px', width: '100%' }}
          />
        </div>
      )}
    </div>
  );
}

