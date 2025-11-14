import React, { useState } from 'react';
import { Info } from 'lucide-react';

export default function InteractiveDeterminantMatrix({ matrix }) {
  const [hoveredCell, setHoveredCell] = useState(null);
  const [selectedRow, setSelectedRow] = useState(null);
  const [selectedCol, setSelectedCol] = useState(null);

  const isSquare = matrix.length > 0 && matrix.length === matrix[0].length;

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
    </div>
  );
}

