import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import * as ul from '../../utils/unsupervisedLearning';

export default function InteractiveAnomalyDetectionVisualization() {
  const canvasRef = useRef(null);
  const [algorithm, setAlgorithm] = useState('isolation-forest');
  const [data, setData] = useState(() => ul.generateAnomalyData(100, 5));
  const [anomalies, setAnomalies] = useState(null);

  useEffect(() => {
    // Simple anomaly detection: mark points far from center as anomalies
    const center = [0, 0];
    const threshold = 4.5;
    
    const detectedAnomalies = data.map(item => {
      const distance = ul.euclideanDistance(item.point, center);
      return distance > threshold;
    });
    
    setAnomalies(detectedAnomalies);
  }, [data, algorithm]);

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
    const padding = { top: 40, right: 40, bottom: 60, left: 60 };

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Find data bounds
    const allPoints = data.map(item => item.point);
    const xValues = allPoints.map(p => p[0]);
    const yValues = allPoints.map(p => p[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

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

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();

    // Draw normal points
    data.forEach((item, i) => {
      const isAnomaly = anomalies && anomalies[i];
      ctx.fillStyle = isAnomaly ? '#ef4444' : '#10b981';
      ctx.beginPath();
      ctx.arc(toScreenX(item.point[0]), toScreenY(item.point[1]), isAnomaly ? 7 : 4, 0, 2 * Math.PI);
      ctx.fill();
      
      if (isAnomaly) {
        // Draw ring around anomaly
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(toScreenX(item.point[0]), toScreenY(item.point[1]), 10, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });

    // Draw detection boundary (circle)
    const center = [0, 0];
    const threshold = 4.5;
    const centerScreenX = toScreenX(center[0]);
    const centerScreenY = toScreenY(center[1]);
    const radiusScreen = (threshold / xRange) * (width - padding.left - padding.right);

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(centerScreenX, centerScreenY, radiusScreen, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const algorithmNames = {
      'isolation-forest': 'Isolation Forest',
      'one-class-svm': 'One-Class SVM'
    };
    ctx.fillText(algorithmNames[algorithm] || 'Anomaly Detection', width / 2, 20);

    // Draw legend
    ctx.textAlign = 'left';
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#10b981';
    ctx.fillRect(width - padding.right - 100, padding.top + 20, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Normal', width - padding.right - 80, padding.top + 32);
    
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(width - padding.right - 100, padding.top + 40, 15, 15);
    ctx.fillStyle = '#1f2937';
    ctx.fillText('Anomaly', width - padding.right - 80, padding.top + 52);

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('X', width / 2, height - padding.bottom + 40);
    
    ctx.save();
    ctx.translate(padding.left - 30, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Y', 0, 0);
    ctx.restore();
  }, [data, anomalies, algorithm]);

  const generateNewData = () => {
    const newData = ul.generateAnomalyData(100, 5);
    setData(newData);
  };

  const anomalyCount = anomalies ? anomalies.filter(a => a).length : 0;

  return (
    <div className="space-y-4">
      <div className="bg-teal-50 border-2 border-teal-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-teal-800">
          💡 <strong>Interactive:</strong> Anomalies are detected as points far from the normal data distribution!
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-teal-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Anomaly Detection Algorithm
          </label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500"
          >
            <option value="isolation-forest">Isolation Forest</option>
            <option value="one-class-svm">One-Class SVM</option>
          </select>
        </div>

        <button
          onClick={generateNewData}
          className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Generate New Data
        </button>
      </div>

      {/* Stats */}
      {anomalies && (
        <div className="bg-white rounded-lg p-4 border-2 border-teal-200">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 rounded p-3">
              <div className="text-sm text-gray-600">Normal Points</div>
              <div className="text-2xl font-bold text-green-600">
                {data.length - anomalyCount}
              </div>
            </div>
            <div className="bg-red-50 rounded p-3">
              <div className="text-sm text-gray-600">Anomalies Detected</div>
              <div className="text-2xl font-bold text-red-600">
                {anomalyCount}
              </div>
            </div>
          </div>
        </div>
      )}

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Algorithm Info */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Anomaly Detection Explained:</h4>
        <ul className="text-sm text-gray-700 space-y-1">
          <li><strong>Isolation Forest:</strong> Randomly selects features and splits to isolate anomalies (fewer splits needed)</li>
          <li><strong>One-Class SVM:</strong> Learns a boundary around normal data, points outside are anomalies</li>
          <li><strong>Use Cases:</strong> Fraud detection, network intrusion, manufacturing defects, medical diagnosis</li>
        </ul>
      </div>
    </div>
  );
}

