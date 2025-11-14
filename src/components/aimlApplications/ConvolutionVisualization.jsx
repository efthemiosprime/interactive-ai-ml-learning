import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function ConvolutionVisualization() {
  const canvasRef = useRef(null);
  const [inputMatrix, setInputMatrix] = useState([]);
  const [kernel, setKernel] = useState([]);
  const [outputMatrix, setOutputMatrix] = useState([]);
  const [step, setStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [kernelType, setKernelType] = useState('edge-detection');

  useEffect(() => {
    initializeMatrices();
  }, [kernelType]);

  useEffect(() => {
    if (inputMatrix.length > 0 && kernel.length > 0) {
      performConvolution();
    }
  }, [inputMatrix, kernel]);

  useEffect(() => {
    drawVisualization();
  }, [inputMatrix, kernel, outputMatrix, step]);

  const initializeMatrices = () => {
    // Initialize 8x8 input matrix (grayscale image)
    const input = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      for (let j = 0; j < 8; j++) {
        // Create a pattern (e.g., edges)
        const value = Math.sin(i * 0.5) * Math.cos(j * 0.5) * 50 + 100;
        row.push(Math.max(0, Math.min(255, value)));
      }
      input.push(row);
    }
    setInputMatrix(input);

    // Initialize kernel based on type
    let newKernel;
    switch (kernelType) {
      case 'edge-detection':
        newKernel = [
          [-1, -1, -1],
          [-1, 8, -1],
          [-1, -1, -1]
        ];
        break;
      case 'blur':
        newKernel = [
          [1/9, 1/9, 1/9],
          [1/9, 1/9, 1/9],
          [1/9, 1/9, 1/9]
        ];
        break;
      case 'sharpen':
        newKernel = [
          [0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]
        ];
        break;
      case 'sobel-x':
        newKernel = [
          [-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]
        ];
        break;
      default:
        newKernel = [
          [-1, -1, -1],
          [-1, 8, -1],
          [-1, -1, -1]
        ];
    }
    setKernel(newKernel);
  };

  const performConvolution = () => {
    if (inputMatrix.length === 0 || kernel.length === 0) return;

    const inputHeight = inputMatrix.length;
    const inputWidth = inputMatrix[0].length;
    const kernelSize = kernel.length;
    const padding = Math.floor(kernelSize / 2);

    const output = [];
    
    // Convolution operation
    for (let i = 0; i < inputHeight - kernelSize + 1; i++) {
      const row = [];
      for (let j = 0; j < inputWidth - kernelSize + 1; j++) {
        let sum = 0;
        
        // Element-wise multiplication and sum
        for (let ki = 0; ki < kernelSize; ki++) {
          for (let kj = 0; kj < kernelSize; kj++) {
            sum += inputMatrix[i + ki][j + kj] * kernel[ki][kj];
          }
        }
        
        row.push(sum);
      }
      output.push(row);
    }

    setOutputMatrix(output);
  };

  const drawVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas || !inputMatrix || inputMatrix.length === 0 || !kernel || kernel.length === 0) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);

    const cellSize = 40;
    const spacing = 10;
    const startX = 50;
    const startY = 50;

    // Draw input matrix
    ctx.fillStyle = '#000';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('Input Matrix (8×8)', startX, startY - 20);
    
    inputMatrix.forEach((row, i) => {
      if (!row) return;
      row.forEach((val, j) => {
        const x = startX + j * (cellSize + spacing);
        const y = startY + i * (cellSize + spacing);
        
        // Color based on value (grayscale)
        const gray = Math.max(0, Math.min(255, val));
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        ctx.fillRect(x, y, cellSize, cellSize);
        
        // Border
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize, cellSize);
        
        // Value
        ctx.fillStyle = gray > 128 ? '#000' : '#fff';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(val).toString(), x + cellSize / 2, y + cellSize / 2 + 3);
      });
    });

    // Draw kernel
    const kernelX = startX + ((inputMatrix[0]?.length || 0) + 1) * (cellSize + spacing) + 50;
    ctx.fillStyle = '#000';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('Kernel (3×3)', kernelX, startY - 20);
    
    kernel.forEach((row, i) => {
      if (!row) return;
      row.forEach((val, j) => {
        const x = kernelX + j * (cellSize + spacing);
        const y = startY + i * (cellSize + spacing);
        
        // Color based on value
        const normalized = (val + 2) / 4; // Normalize to 0-1
        const gray = Math.max(0, Math.min(255, normalized * 255));
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        ctx.fillRect(x, y, cellSize, cellSize);
        
        // Border
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellSize, cellSize);
        
        // Value
        ctx.fillStyle = normalized > 0.5 ? '#000' : '#fff';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2 + 4);
      });
    });

    // Draw arrow
    const arrowX = kernelX - 30;
    const arrowY = startY + kernel.length * (cellSize + spacing) / 2;
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 20, arrowY);
    ctx.lineTo(arrowX - 15, arrowY - 5);
    ctx.moveTo(arrowX - 20, arrowY);
    ctx.lineTo(arrowX - 15, arrowY + 5);
    ctx.stroke();

    // Draw output matrix
    if (!outputMatrix || outputMatrix.length === 0) return;
    
    const outputX = startX;
    const outputY = startY + (inputMatrix.length + 1) * (cellSize + spacing) + 30;
    ctx.fillStyle = '#000';
    ctx.font = 'bold 14px Arial';
    ctx.fillText(`Output Matrix (${outputMatrix.length}×${outputMatrix[0]?.length || 0})`, outputX, outputY - 20);
    
    outputMatrix.forEach((row, i) => {
      if (!row) return;
      row.forEach((val, j) => {
        const x = outputX + j * (cellSize + spacing);
        const y = outputY + i * (cellSize + spacing);
        
        // Color based on value
        const normalized = (val + 500) / 1000; // Normalize
        const gray = Math.max(0, Math.min(255, normalized * 255));
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        ctx.fillRect(x, y, cellSize, cellSize);
        
        // Border
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellSize, cellSize);
        
        // Value
        ctx.fillStyle = normalized > 0.5 ? '#000' : '#fff';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(val).toString(), x + cellSize / 2, y + cellSize / 2 + 3);
      });
    });

    // Draw step-by-step visualization if animating
    if (isAnimating && outputMatrix.length > 0 && outputMatrix[0] && step < outputMatrix.length * outputMatrix[0].length) {
      const currentRow = Math.floor(step / outputMatrix[0].length);
      const currentCol = step % outputMatrix[0].length;
      
      if (currentRow < outputMatrix.length && currentCol < outputMatrix[0].length) {
        // Highlight input region
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 3;
        const highlightX = startX + currentCol * (cellSize + spacing);
        const highlightY = startY + currentRow * (cellSize + spacing);
        ctx.strokeRect(highlightX - 2, highlightY - 2, kernel.length * (cellSize + spacing) + 4, kernel.length * (cellSize + spacing) + 4);
      }
    }
  };

  const animateConvolution = async () => {
    if (!outputMatrix || outputMatrix.length === 0 || !outputMatrix[0]) {
      return;
    }
    
    setIsAnimating(true);
    setStep(0);
    
    const totalSteps = outputMatrix.length * outputMatrix[0].length;
    
    for (let i = 0; i < totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, 200));
      setStep(i);
    }
    
    setIsAnimating(false);
  };

  const codeExample = `# Convolution Operation in CNNs

import numpy as np
import torch
import torch.nn as nn

def convolution_manual(input_matrix, kernel):
    """
    Manual Convolution Operation
    
    Formula: (I * K)[i,j] = Σ Σ I[i+m, j+n] · K[m, n]
    
    where:
    - I = input matrix
    - K = kernel/filter
    - * = convolution operator
    """
    input_height, input_width = input_matrix.shape
    kernel_size = kernel.shape[0]
    
    output_height = input_height - kernel_size + 1
    output_width = input_width - kernel_size + 1
    
    output = np.zeros((output_height, output_width))
    
    # Slide kernel over input
    for i in range(output_height):
        for j in range(output_width):
            # Element-wise multiplication and sum
            sum_val = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    sum_val += input_matrix[i + m, j + n] * kernel[m, n]
            output[i, j] = sum_val
    
    return output

# Example: Edge detection
input_image = np.array([
    [100, 100, 100, 100],
    [100, 200, 200, 100],
    [100, 200, 200, 100],
    [100, 100, 100, 100]
])

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Perform convolution
output = convolution_manual(input_image, edge_kernel)
print("Input shape:", input_image.shape)
print("Kernel shape:", edge_kernel.shape)
print("Output shape:", output.shape)
print("Output:")
print(output)

# Using PyTorch (for CNNs)
# Convolution layer automatically performs this operation
conv_layer = nn.Conv2d(
    in_channels=1,      # Input channels (grayscale)
    out_channels=1,     # Output channels
    kernel_size=3,      # 3x3 kernel
    stride=1,          # Step size
    padding=0           # Padding
)

# Set kernel weights
with torch.no_grad():
    conv_layer.weight[0, 0] = torch.tensor(edge_kernel, dtype=torch.float32)

# Convert input to tensor format (batch, channels, height, width)
input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Forward pass
output_tensor = conv_layer(input_tensor)
print("PyTorch output:")
print(output_tensor.squeeze().numpy())

# Multiple kernels (feature maps)
conv_multi = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
output_multi = conv_multi(input_tensor)
print("Multiple kernels output shape:", output_multi.shape)
# Shape: (batch=1, channels=4, height=2, width=2)`;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200 mb-4">
        <h2 className="text-xl font-bold text-purple-900 mb-2">Convolution Operation Visualization</h2>
        <p className="text-purple-800 text-sm">
          Understand how matrix operations work in Convolutional Neural Networks (CNNs). See the convolution operation step-by-step.
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Kernel Type
            </label>
            <select
              value={kernelType}
              onChange={(e) => {
                setKernelType(e.target.value);
                setStep(0);
                setIsAnimating(false);
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            >
              <option value="edge-detection">Edge Detection</option>
              <option value="blur">Blur</option>
              <option value="sharpen">Sharpen</option>
              <option value="sobel-x">Sobel X (Horizontal Edges)</option>
            </select>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={animateConvolution}
              disabled={isAnimating}
              className="flex-1 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {isAnimating ? 'Animating...' : 'Animate Convolution'}
            </button>
            <button
              onClick={() => {
                setStep(0);
                setIsAnimating(false);
              }}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Kernel Explanation */}
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">Current Kernel:</h4>
          <div className="text-sm text-blue-800">
            {kernelType === 'edge-detection' && 'Detects edges by highlighting areas where pixel values change rapidly.'}
            {kernelType === 'blur' && 'Averages neighboring pixels to create a smoothing/blurring effect.'}
            {kernelType === 'sharpen' && 'Enhances edges and details by emphasizing differences from neighbors.'}
            {kernelType === 'sobel-x' && 'Detects horizontal edges using gradient approximation.'}
          </div>
        </div>
      </div>

      {/* Canvas Visualization */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Convolution Operation</h3>
        <div className="flex justify-center overflow-x-auto">
          <canvas
            ref={canvasRef}
            width={1000}
            height={700}
            className="border-2 border-gray-300 rounded-lg"
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-800 border border-gray-600"></div>
            <span>Input Matrix (8×8)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-600 border-2 border-gray-800"></div>
            <span>Kernel (3×3)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 border-2 border-green-700"></div>
            <span>Output Matrix (6×6)</span>
          </div>
          {isAnimating && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-orange-500 border-2 border-orange-700"></div>
              <span>Current Region</span>
            </div>
          )}
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white rounded-lg p-4 shadow-md">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Convolution Implementation</h3>
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            customStyle={{ margin: 0, borderRadius: '0.5rem' }}
            showLineNumbers
          >
            {codeExample}
          </SyntaxHighlighter>
        </div>
      </div>

      {/* Mathematical Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Mathematical Foundations:</h3>
        <div className="space-y-2 text-blue-800 text-sm">
          <div>
            <strong>Linear Algebra - Matrix Operations:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Convolution: (I * K)[i,j] = Σ Σ I[i+m, j+n] · K[m, n]</li>
              <li>Element-wise multiplication and summation</li>
              <li>Sliding window operation across input</li>
            </ul>
          </div>
          <div>
            <strong>How It Works:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Kernel slides over input matrix</li>
              <li>At each position: multiply overlapping elements</li>
              <li>Sum all products to get output value</li>
              <li>Output size = Input size - Kernel size + 1</li>
            </ul>
          </div>
          <div>
            <strong>In CNNs:</strong>
            <ul className="ml-4 mt-1 list-disc list-inside">
              <li>Kernels are learned weights (filters)</li>
              <li>Each kernel detects different features</li>
              <li>Multiple kernels create feature maps</li>
              <li>Foundation of image processing in deep learning</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

