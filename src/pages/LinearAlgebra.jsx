import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import LinearAlgebraControls from '../components/linearAlgebra/LinearAlgebraControls';
import LinearAlgebraVisualization from '../components/linearAlgebra/LinearAlgebraVisualization';
import LinearAlgebraEducationalPanels from '../components/linearAlgebra/LinearAlgebraEducationalPanels';

export default function LinearAlgebra() {
  const [selectedTopic, setSelectedTopic] = useState('eigenvalues');
  const [matrix, setMatrix] = useState([[2, 1], [1, 2]]);
  const [matrixA, setMatrixA] = useState([[1, 2], [3, 4]]);
  const [matrixB, setMatrixB] = useState([[5, 6], [7, 8]]);
  const [matrixSizeA, setMatrixSizeA] = useState({ rows: 2, cols: 2 });
  const [matrixSizeB, setMatrixSizeB] = useState({ rows: 2, cols: 2 });

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      <div className="max-w-7xl mx-auto px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-indigo-600 hover:text-indigo-800 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </Link>
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">
            Linear Algebra for AI & ML
          </h1>
          <p className="text-gray-700 mt-2">
            Master eigenvalues, eigenvectors, and understand how data and weights are represented in ML models
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Controls Panel */}
          <div className="lg:col-span-1">
            <LinearAlgebraControls
              selectedTopic={selectedTopic}
              setSelectedTopic={setSelectedTopic}
              matrix={matrix}
              setMatrix={setMatrix}
              matrixA={matrixA}
              setMatrixA={setMatrixA}
              matrixB={matrixB}
              setMatrixB={setMatrixB}
              matrixSizeA={matrixSizeA}
              setMatrixSizeA={setMatrixSizeA}
              matrixSizeB={matrixSizeB}
              setMatrixSizeB={setMatrixSizeB}
            />
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <LinearAlgebraVisualization
              selectedTopic={selectedTopic}
              matrix={matrix}
              matrixA={matrixA}
              matrixB={matrixB}
              matrixSizeA={matrixSizeA}
              matrixSizeB={matrixSizeB}
            />
          </div>
        </div>

        {/* Educational Panels */}
        <div className="mt-8">
          <LinearAlgebraEducationalPanels
            selectedTopic={selectedTopic}
            matrix={matrix}
            matrixA={matrixA}
            matrixB={matrixB}
          />
        </div>
      </div>
    </div>
  );
}

