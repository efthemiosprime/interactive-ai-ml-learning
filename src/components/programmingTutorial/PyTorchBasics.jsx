import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function PyTorchBasics() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const codeExamples = [
    {
      title: '1. Creating Tensors',
      code: `import torch

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Create tensor with specific shape
z = torch.zeros(3, 4)
ones = torch.ones(2, 3)

# Random tensor
random_tensor = torch.randn(3, 3)

print("x:", x)
print("y:", y)
print("z:", z)
print("random_tensor:", random_tensor)`,
      output: `x: tensor([1., 2., 3.])
y: tensor([[1., 2.],
           [3., 4.]])
z: tensor([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
random_tensor: tensor([[-0.2341,  0.5678, -0.1234],
                       [ 0.8901, -0.3456,  0.6789],
                       [-0.4567,  0.2345, -0.7890]])`
    },
    {
      title: '2. Tensor Operations',
      code: `import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
add = a + b
multiply = a * b
power = a ** 2

# Matrix multiplication
matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
matmul = torch.matmul(matrix_a, matrix_b)

print("a + b:", add)
print("a * b:", multiply)
print("a^2:", power)
print("Matrix multiplication:", matmul)`,
      output: `a + b: tensor([5., 7., 9.])
a * b: tensor([ 4., 10., 18.])
a^2: tensor([1., 4., 9.])
Matrix multiplication: tensor([[19., 22.],
                                [43., 50.]])`
    },
    {
      title: '3. Automatic Differentiation',
      code: `import torch

# Create tensor with requires_grad=True for gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define a function: z = x^2 + y^2
z = x**2 + y**2

# Compute gradients
z.backward()

print("x:", x.item())
print("y:", y.item())
print("z:", z.item())
print("dz/dx:", x.grad.item())
print("dz/dy:", y.grad.item())`,
      output: `x: 2.0
y: 3.0
z: 13.0
dz/dx: 4.0
dz/dy: 6.0`
    },
    {
      title: '4. Simple Linear Model',
      code: `import torch
import torch.nn as nn

# Define a simple linear model
model = nn.Linear(3, 1)  # 3 inputs, 1 output

# Create sample input
x = torch.tensor([[1.0, 2.0, 3.0]])

# Forward pass
output = model(x)

print("Input:", x)
print("Output:", output)
print("Model weights:", model.weight.data)
print("Model bias:", model.bias.data)`,
      output: `Input: tensor([[1., 2., 3.]])
Output: tensor([[0.2341]], grad_fn=<AddmmBackward0>)
Model weights: tensor([[-0.1234,  0.5678, -0.2345]])
Model bias: tensor([0.1234])`
    }
  ];

  const [selectedExample, setSelectedExample] = useState(0);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const runCode = () => {
    setOutput(codeExamples[selectedExample].output);
  };

  return (
    <div className="space-y-4">
      <div className="bg-orange-50 border-2 border-orange-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-orange-800">
          💡 <strong>Interactive:</strong> Explore PyTorch basics with code examples. Click "Run Code" to see outputs!
        </p>
      </div>

      {/* Example Selector */}
      <div className="bg-white rounded-lg p-4 border-2 border-orange-200">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Example
        </label>
        <select
          value={selectedExample}
          onChange={(e) => {
            setSelectedExample(parseInt(e.target.value));
            setOutput('');
          }}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500"
        >
          {codeExamples.map((ex, idx) => (
            <option key={idx} value={idx}>{ex.title}</option>
          ))}
        </select>
      </div>

      {/* Code Display */}
      <div className="bg-gray-900 rounded-lg p-4 relative">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400 text-sm">Python</span>
          <button
            onClick={() => copyToClipboard(codeExamples[selectedExample].code)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
          </button>
        </div>
        <pre className="text-green-400 text-sm overflow-x-auto">
          <code>{codeExamples[selectedExample].code}</code>
        </pre>
      </div>

      {/* Run Button */}
      <button
        onClick={runCode}
        className="w-full px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-semibold flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Run Code
      </button>

      {/* Output Display */}
      {output && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-2">Output</div>
          <pre className="text-green-400 text-sm whitespace-pre-wrap">
            {output}
          </pre>
        </div>
      )}

      {/* Explanation */}
      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">Explanation</h4>
        {selectedExample === 0 && (
          <p className="text-sm text-blue-800">
            Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with GPU support.
            Use <code className="bg-blue-100 px-1 rounded">torch.tensor()</code> to create tensors from Python lists,
            or use helper functions like <code className="bg-blue-100 px-1 rounded">torch.zeros()</code>, 
            <code className="bg-blue-100 px-1 rounded">torch.ones()</code>, and <code className="bg-blue-100 px-1 rounded">torch.randn()</code>.
          </p>
        )}
        {selectedExample === 1 && (
          <p className="text-sm text-blue-800">
            PyTorch supports element-wise operations (+, -, *, /) and matrix operations. Use 
            <code className="bg-blue-100 px-1 rounded">torch.matmul()</code> for matrix multiplication.
            Operations are automatically broadcasted when shapes are compatible.
          </p>
        )}
        {selectedExample === 2 && (
          <p className="text-sm text-blue-800">
            Automatic differentiation is PyTorch's key feature. Set <code className="bg-blue-100 px-1 rounded">requires_grad=True</code>
            to track operations. Call <code className="bg-blue-100 px-1 rounded">.backward()</code> to compute gradients,
            which are stored in the <code className="bg-blue-100 px-1 rounded">.grad</code> attribute.
          </p>
        )}
        {selectedExample === 3 && (
          <p className="text-sm text-blue-800">
            <code className="bg-blue-100 px-1 rounded">nn.Linear</code> creates a linear layer (fully connected layer).
            It applies the transformation: <code className="bg-blue-100 px-1 rounded">output = input × weight^T + bias</code>.
            This is the building block for neural networks.
          </p>
        )}
      </div>
    </div>
  );
}

