import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function TensorFlowBasics() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const codeExamples = [
    {
      title: '1. Creating Tensors',
      code: `import tensorflow as tf

# Create tensors
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# Create tensor with specific shape
z = tf.zeros((3, 4))
ones = tf.ones((2, 3))

# Random tensor
random_tensor = tf.random.normal((3, 3))

print("x:", x.numpy())
print("y:", y.numpy())
print("z:", z.numpy())
print("random_tensor:", random_tensor.numpy())`,
      output: `x: [1. 2. 3.]
y: [[1. 2.]
     [3. 4.]]
z: [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
random_tensor: [[-0.2341  0.5678 -0.1234]
                 [ 0.8901 -0.3456  0.6789]
                 [-0.4567  0.2345 -0.7890]]`
    },
    {
      title: '2. Tensor Operations',
      code: `import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# Element-wise operations
add = a + b
multiply = a * b
power = tf.pow(a, 2)

# Matrix multiplication
matrix_a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
matrix_b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
matmul = tf.matmul(matrix_a, matrix_b)

print("a + b:", add.numpy())
print("a * b:", multiply.numpy())
print("a^2:", power.numpy())
print("Matrix multiplication:", matmul.numpy())`,
      output: `a + b: [5. 7. 9.]
a * b: [ 4. 10. 18.]
a^2: [1. 4. 9.]
Matrix multiplication: [[19. 22.]
                          [43. 50.]]`
    },
    {
      title: '3. Automatic Differentiation',
      code: `import tensorflow as tf

# Create variables for gradient tracking
x = tf.Variable(2.0)
y = tf.Variable(3.0)

# Use GradientTape to record operations
with tf.GradientTape() as tape:
    # Define a function: z = x^2 + y^2
    z = x**2 + y**2

# Compute gradients
gradients = tape.gradient(z, [x, y])

print("x:", x.numpy())
print("y:", y.numpy())
print("z:", z.numpy())
print("dz/dx:", gradients[0].numpy())
print("dz/dy:", gradients[1].numpy())`,
      output: `x: 2.0
y: 3.0
z: 13.0
dz/dx: 4.0
dz/dy: 6.0`
    },
    {
      title: '4. Simple Linear Model with Keras',
      code: `import tensorflow as tf
from tensorflow import keras

# Define a simple linear model using Keras
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(3,))  # 3 inputs, 1 output
])

# Create sample input
x = tf.constant([[1.0, 2.0, 3.0]])

# Forward pass
output = model(x)

print("Input:", x.numpy())
print("Output:", output.numpy())
print("Model weights:", model.layers[0].get_weights()[0])
print("Model bias:", model.layers[0].get_weights()[1])`,
      output: `Input: [[1. 2. 3.]]
Output: [[0.2341]]
Model weights: [[-0.1234]
                 [ 0.5678]
                 [-0.2345]]
Model bias: [0.1234]`
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
          💡 <strong>Interactive:</strong> Explore TensorFlow basics with code examples. Click "Run Code" to see outputs!
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
            TensorFlow uses <code className="bg-blue-100 px-1 rounded">tf.constant()</code> to create immutable tensors.
            Use <code className="bg-blue-100 px-1 rounded">tf.zeros()</code>, <code className="bg-blue-100 px-1 rounded">tf.ones()</code>,
            and <code className="bg-blue-100 px-1 rounded">tf.random.normal()</code> for common tensor creation.
            Call <code className="bg-blue-100 px-1 rounded">.numpy()</code> to convert to NumPy arrays.
          </p>
        )}
        {selectedExample === 1 && (
          <p className="text-sm text-blue-800">
            TensorFlow supports element-wise operations and matrix operations. Use 
            <code className="bg-blue-100 px-1 rounded">tf.matmul()</code> for matrix multiplication.
            Operations follow NumPy broadcasting rules.
          </p>
        )}
        {selectedExample === 2 && (
          <p className="text-sm text-blue-800">
            TensorFlow uses <code className="bg-blue-100 px-1 rounded">tf.GradientTape()</code> for automatic differentiation.
            Operations inside the tape context are recorded. Call <code className="bg-blue-100 px-1 rounded">tape.gradient()</code>
            to compute gradients with respect to variables.
          </p>
        )}
        {selectedExample === 3 && (
          <p className="text-sm text-blue-800">
            Keras provides a high-level API for building models. <code className="bg-blue-100 px-1 rounded">keras.layers.Dense</code>
            creates a fully connected layer. The model applies: <code className="bg-blue-100 px-1 rounded">output = activation(input × weight + bias)</code>.
          </p>
        )}
      </div>
    </div>
  );
}

