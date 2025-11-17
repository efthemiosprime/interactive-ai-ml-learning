import React, { useState, useRef, useEffect } from 'react';
import { Info, X } from 'lucide-react';

export default function InteractiveDataRepresentation() {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const canvasRef = useRef(null);

  // Example data matrix: 4 samples, 3 features
  const dataMatrix = [
    [25, 50000, 2],   // Sample 1: Age, Income, Purchases
    [35, 75000, 5],   // Sample 2
    [28, 60000, 3],   // Sample 3
    [45, 90000, 8],   // Sample 4
  ];

  const features = ['Age', 'Income ($)', 'Purchases'];
  const samples = ['Customer 1', 'Customer 2', 'Customer 3', 'Customer 4'];

  // Normalize data for visualization (0-1 scale)
  const normalizeData = (data) => {
    const normalized = data.map(row => [...row]);
    for (let col = 0; col < normalized[0].length; col++) {
      const columnValues = normalized.map(row => row[col]);
      const min = Math.min(...columnValues);
      const max = Math.max(...columnValues);
      const range = max - min || 1;
      for (let row = 0; row < normalized.length; row++) {
        normalized[row][col] = (normalized[row][col] - min) / range;
      }
    }
    return normalized;
  };

  const normalizedData = normalizeData(dataMatrix);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 800;
    const displayHeight = rect.height > 0 ? rect.height : 400;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (i / 10) * chartWidth;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    for (let i = 0; i <= 10; i++) {
      const y = padding.top + (i / 10) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Feature Index', width / 2, height - 10);
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Sample Index', 0, 0);
    ctx.restore();

    // Draw data points as heatmap
    const cellWidth = chartWidth / features.length;
    const cellHeight = chartHeight / dataMatrix.length;
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

    dataMatrix.forEach((row, rowIdx) => {
      row.forEach((val, colIdx) => {
        const x = padding.left + colIdx * cellWidth;
        const y = padding.top + rowIdx * cellHeight;
        const normalizedVal = normalizedData[rowIdx][colIdx];
        
        // Color intensity based on normalized value
        const intensity = Math.floor(normalizedVal * 255);
        const color = `rgba(59, 130, 246, ${0.3 + normalizedVal * 0.7})`;
        
        // Highlight selected sample
        if (selectedSample === rowIdx) {
          ctx.fillStyle = colors[rowIdx % colors.length];
          ctx.globalAlpha = 0.3;
          ctx.fillRect(x, y, cellWidth, cellHeight);
          ctx.globalAlpha = 1.0;
        }

        // Draw cell
        ctx.fillStyle = color;
        ctx.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);

        // Draw border
        ctx.strokeStyle = selectedSample === rowIdx ? colors[rowIdx % colors.length] : '#d1d5db';
        ctx.lineWidth = selectedSample === rowIdx ? 2 : 1;
        ctx.strokeRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);

        // Draw value
        ctx.fillStyle = '#1f2937';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
          val.toLocaleString(),
          x + cellWidth / 2,
          y + cellHeight / 2 + 3
        );
      });
    });

    // Draw feature labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    features.forEach((feature, idx) => {
      ctx.fillText(
        feature,
        padding.left + idx * cellWidth + cellWidth / 2,
        padding.top - 10
      );
    });

    // Draw sample labels
    ctx.textAlign = 'right';
    samples.forEach((sample, idx) => {
      ctx.fillText(
        sample,
        padding.left - 10,
        padding.top + idx * cellHeight + cellHeight / 2 + 3
      );
    });

    // Draw legend
    ctx.fillStyle = '#1f2937';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Intensity = Feature Value', padding.left, height - 30);
    
    // Color scale
    const scaleWidth = 200;
    const scaleX = width - padding.right - scaleWidth;
    const scaleY = height - 30;
    const gradient = ctx.createLinearGradient(scaleX, scaleY, scaleX + scaleWidth, scaleY);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 1.0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(scaleX, scaleY, scaleWidth, 10);
    ctx.strokeStyle = '#6b7280';
    ctx.strokeRect(scaleX, scaleY, scaleWidth, 10);
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px sans-serif';
    ctx.fillText('Low', scaleX - 20, scaleY + 8);
    ctx.fillText('High', scaleX + scaleWidth + 5, scaleY + 8);
  }, [selectedSample, dataMatrix, features, samples, normalizedData]);

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-6 border-2 border-indigo-200">
        <h4 className="font-bold text-indigo-900 mb-4">Interactive Data Matrix</h4>
        <p className="text-sm text-gray-700 mb-4">
          Hover over cells to see details. Click a row to highlight the entire sample.
        </p>

        {/* Interactive Matrix */}
        <div className="bg-white rounded-lg p-4 border-2 border-indigo-300 overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Header Row */}
            <div className="flex gap-2 mb-2">
              <div className="w-24 flex-shrink-0"></div>
              {features.map((feature, idx) => (
                <div
                  key={idx}
                  className="w-24 text-center px-2 py-1 bg-indigo-100 text-indigo-900 font-semibold rounded text-sm"
                >
                  {feature}
                </div>
              ))}
            </div>

            {/* Data Rows */}
            {dataMatrix.map((row, i) => (
              <div
                key={i}
                className={`flex gap-2 mb-1 transition-all cursor-pointer ${
                  selectedSample === i ? 'bg-yellow-100 rounded px-2 py-1' : ''
                }`}
                onClick={() => setSelectedSample(selectedSample === i ? null : i)}
              >
                <div className="w-24 flex-shrink-0 text-sm font-semibold text-gray-700 flex items-center">
                  {samples[i]}
                </div>
                {row.map((val, j) => {
                  const cellKey = `${i}-${j}`;
                  const isHovered = hoveredCell === cellKey;
                  return (
                    <div
                      key={j}
                      onMouseEnter={() => setHoveredCell(cellKey)}
                      onMouseLeave={() => setHoveredCell(null)}
                      className={`w-24 text-center px-2 py-2 rounded transition-all cursor-pointer ${
                        isHovered
                          ? 'bg-indigo-600 text-white font-bold scale-110 ring-2 ring-yellow-400 z-10 relative'
                          : selectedSample === i
                          ? 'bg-yellow-200 text-gray-900 font-semibold'
                          : 'bg-gray-50 text-gray-700 hover:bg-indigo-100'
                      }`}
                    >
                      {val.toLocaleString()}
                      {isHovered && (
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap z-20">
                          Sample {i + 1}, Feature {j + 1}: {val}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Selected Sample Info */}
        {selectedSample !== null && (
          <div className="mt-4 bg-yellow-50 border-2 border-yellow-300 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <h5 className="font-bold text-yellow-900 mb-2">
                  {samples[selectedSample]} - Sample Details
                </h5>
                <div className="space-y-1 text-sm text-yellow-800">
                  {features.map((feature, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="font-semibold">{feature}:</span>
                      <span>{dataMatrix[selectedSample][idx].toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              <button
                onClick={() => setSelectedSample(null)}
                className="text-yellow-700 hover:text-yellow-900"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* Canvas Visualization */}
        <div className="mt-6 bg-white rounded-lg p-4 border-2 border-indigo-300">
          <h5 className="font-bold text-indigo-900 mb-3 text-center">Data Matrix Heatmap Visualization</h5>
          <p className="text-xs text-gray-600 mb-3 text-center">
            Visual representation of the data matrix. Color intensity represents feature values. Click a row above to highlight it.
          </p>
          <canvas
            ref={canvasRef}
            className="w-full border border-gray-300 rounded-lg"
            style={{ height: '400px', width: '100%' }}
          />
          <div className="mt-3 flex items-center justify-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-indigo-200 border border-indigo-400"></div>
              <span>Low value</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-indigo-600 border border-indigo-800"></div>
              <span>High value</span>
            </div>
            {selectedSample !== null && (
              <div className="flex items-center gap-2">
                <div className={`w-4 h-4 border-2 ${['bg-blue-200', 'bg-green-200', 'bg-yellow-200', 'bg-red-200'][selectedSample]}`}></div>
                <span>Selected sample</span>
              </div>
            )}
          </div>
        </div>

        {/* ML Context */}
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-blue-900">
              <strong>In ML:</strong> This matrix would be fed to a model. Each row (sample) contains 
              features that the model uses to make predictions. The model learns patterns across all samples.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

