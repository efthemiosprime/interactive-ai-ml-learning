import React, { useState } from 'react';
import { Brain, Network, Image, MessageSquare, TrendingUp, Layers, Zap, Target, BarChart, Users, CreditCard, Settings, Shield, FileText, Cpu, Database, Activity, Gauge, Rocket } from 'lucide-react';
import { ChevronDown, ChevronUp } from 'lucide-react';

export default function MLUseCasesPanel({ domain, operationType }) {
  const [expandedCase, setExpandedCase] = useState(null);

  const useCasesByDomain = {
    'calculus': {
      'derivatives': [
        {
          title: 'Loss Function Optimization',
          icon: <Target className="w-6 h-6" />,
          description: 'Derivatives find where loss function is minimized',
          stepByStep: [
            'Start with a loss function L(θ) that measures prediction error',
            'Calculate derivative dL/dθ to find slope',
            'Move in direction opposite to gradient: θ = θ - α × dL/dθ',
            'Repeat until derivative ≈ 0 (minimum found)'
          ],
          example: 'Mean Squared Error: L = (1/n)Σ(y_pred - y_true)²',
          code: `# Gradient descent for linear regression
def loss_function(theta, X, y):
    predictions = X @ theta
    return np.mean((predictions - y)**2)

# Calculate gradient
def gradient(theta, X, y):
    predictions = X @ theta
    return (2/len(y)) * X.T @ (predictions - y)

# Update parameters
theta = theta - learning_rate * gradient(theta, X, y)`,
          realWorld: 'Training all ML models - linear regression, neural networks, SVM',
          visual: 'Shows loss curve decreasing as we move down the gradient'
        },
        {
          title: 'Learning Rate Tuning',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Derivative magnitude determines optimal step size',
          stepByStep: [
            'Large derivative = steep slope = take bigger steps',
            'Small derivative = gentle slope = take smaller steps',
            'Adaptive learning rates adjust based on derivative magnitude',
            'Prevents overshooting minimum or moving too slowly'
          ],
          example: 'Adam optimizer adapts learning rate using gradient history',
          code: `# Adaptive learning rate (Adam optimizer)
m = beta1 * m + (1 - beta1) * gradient  # Momentum
v = beta2 * v + (1 - beta2) * gradient**2  # RMSprop
theta = theta - lr * m / (sqrt(v) + eps)`,
          realWorld: 'All deep learning frameworks - PyTorch, TensorFlow, Keras',
          visual: 'Shows how step size adapts to terrain steepness'
        },
        {
          title: 'Hyperparameter Optimization',
          icon: <Settings className="w-6 h-6" />,
          description: 'Derivatives help find optimal hyperparameters like learning rate, regularization strength',
          stepByStep: [
            'Define hyperparameter search space',
            'Use derivatives to estimate sensitivity: ∂L/∂hyperparameter',
            'Large derivative = hyperparameter has big impact',
            'Adjust hyperparameters based on derivative magnitude',
            'Use grid search or Bayesian optimization guided by derivatives'
          ],
          example: 'Learning rate sensitivity: ∂L/∂lr shows how much loss changes with learning rate',
          code: `# Hyperparameter sensitivity using derivatives
def hyperparameter_sensitivity(model, hyperparam, data):
    # Compute loss with small change in hyperparameter
    loss1 = train_and_evaluate(model, hyperparam, data)
    loss2 = train_and_evaluate(model, hyperparam + epsilon, data)
    sensitivity = (loss2 - loss1) / epsilon
    return sensitivity  # Large = sensitive, small = insensitive`,
          realWorld: 'AutoML systems, neural architecture search, hyperparameter tuning tools',
          visual: 'Shows loss surface as function of hyperparameters with optimal region'
        },
        {
          title: 'Regularization Strength Tuning',
          icon: <Shield className="w-6 h-6" />,
          description: 'Derivatives help balance model complexity and overfitting',
          stepByStep: [
            'Loss = Data Loss + λ × Regularization',
            'Compute derivative: ∂L/∂λ to see regularization impact',
            'Large derivative = regularization has strong effect',
            'Adjust λ to balance fit and generalization',
            'Find λ where derivative indicates good trade-off'
          ],
          example: 'L2 regularization: L = MSE + λ||w||², ∂L/∂λ = ||w||² shows regularization contribution',
          code: `# Regularization tuning
def compute_regularization_gradient(weights, lambda_reg):
    data_loss_grad = compute_data_loss_gradient()
    reg_grad = lambda_reg * 2 * weights  # L2 regularization
    total_grad = data_loss_grad + reg_grad
    # Adjust lambda based on gradient balance
    return total_grad`,
          realWorld: 'Preventing overfitting, model selection, cross-validation',
          visual: 'Shows bias-variance trade-off curve with optimal regularization point'
        },
        {
          title: 'Convergence Detection',
          icon: <Gauge className="w-6 h-6" />,
          description: 'Derivative magnitude indicates when optimization has converged',
          stepByStep: [
            'Monitor derivative magnitude during training',
            'When |dL/dθ| ≈ 0, we\'re near a minimum',
            'Stop training when derivative is small enough',
            'Prevents unnecessary computation',
            'Indicates optimal parameters found'
          ],
          example: 'If |dL/dθ| < 1e-6, we\'ve converged - loss won\'t improve much more',
          code: `# Convergence detection
while training:
    gradient = compute_gradient(theta, X, y)
    theta = theta - learning_rate * gradient
    
    # Check convergence
    if np.linalg.norm(gradient) < 1e-6:
        print("Converged!")
        break`,
          realWorld: 'Early stopping, efficient training, resource optimization',
          visual: 'Shows gradient magnitude decreasing to zero as training progresses'
        }
      ],
      'partial-derivatives': [
        {
          title: 'Multi-Parameter Neural Networks',
          icon: <Network className="w-6 h-6" />,
          description: 'Each weight has its own partial derivative, computed independently',
          stepByStep: [
            'Neural network has thousands/millions of weights',
            'Compute ∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ separately',
            'Each partial derivative shows how one weight affects loss',
            'Update each weight independently: wᵢ = wᵢ - α × ∂L/∂wᵢ',
            'All weights updated simultaneously using their partial derivatives'
          ],
          example: 'A layer with 784×128 weights needs 100,352 partial derivatives computed',
          code: `# Computing partial derivatives for each weight
for i in range(num_weights):
    # Compute partial derivative for weight i
    partial_deriv = compute_partial_derivative(loss, weights[i])
    # Update this weight independently
    weights[i] = weights[i] - learning_rate * partial_deriv`,
          realWorld: 'Training all neural networks - each weight optimized independently',
          visual: 'Shows individual weight updates based on their partial derivatives'
        },
        {
          title: 'Feature Importance Analysis',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Partial derivatives reveal which features most impact predictions',
          stepByStep: [
            'Compute ∂L/∂feature₁, ∂L/∂feature₂, ..., ∂L/∂featureₙ',
            'Large partial derivative = feature has big impact',
            'Small partial derivative = feature has little impact',
            'Use for feature selection and interpretation',
            'Remove features with near-zero partial derivatives'
          ],
          example: 'If ∂L/∂age = 0.5 and ∂L/∂income = 0.01, age is more important than income',
          code: `# Feature importance using partial derivatives
feature_importance = {}
for feature in features:
    partial_deriv = compute_partial_derivative(loss, feature)
    feature_importance[feature] = abs(partial_deriv)

# Sort by importance
important_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)`,
          realWorld: 'Feature selection, model interpretability, explainable AI',
          visual: 'Shows bar chart of feature importances based on partial derivative magnitudes'
        },
        {
          title: 'Gradient Vector Construction',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Gradients are vectors of partial derivatives - one per parameter',
          stepByStep: [
            'Compute partial derivative for each parameter',
            'Collect all partial derivatives into a vector',
            'Gradient = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]',
            'This gradient vector guides optimization',
            'Each component is a partial derivative'
          ],
          example: 'For 3 parameters: ∇L = [∂L/∂w₁, ∂L/∂w₂, ∂L/∂w₃]',
          code: `# Constructing gradient from partial derivatives
def compute_gradient(weights, X, y):
    gradient = []
    for i in range(len(weights)):
        # Compute partial derivative for weight i
        partial = compute_partial_derivative(loss, weights[i], X, y)
        gradient.append(partial)
    return np.array(gradient)  # Gradient vector`,
          realWorld: 'All gradient-based optimization - SGD, Adam, RMSprop',
          visual: 'Shows how individual partial derivatives combine into gradient vector'
        },
        {
          title: 'Multi-Output Models',
          icon: <Target className="w-6 h-6" />,
          description: 'Partial derivatives handle models with multiple outputs',
          stepByStep: [
            'Model predicts multiple outputs: [y₁, y₂, ..., yₖ]',
            'Compute ∂L/∂output₁, ∂L/∂output₂, ..., ∂L/∂outputₖ',
            'Each output has its own partial derivative',
            'Update model to optimize all outputs simultaneously',
            'Balance improvements across all outputs'
          ],
          example: 'Multi-task learning: predict age, income, and purchase probability simultaneously',
          code: `# Multi-output model with partial derivatives
def multi_output_loss(predictions, targets):
    loss1 = mse_loss(predictions[0], targets[0])  # Age prediction
    loss2 = mse_loss(predictions[1], targets[1])  # Income prediction
    loss3 = bce_loss(predictions[2], targets[2])  # Purchase prediction
    
    # Partial derivatives for each output
    dL_doutput1 = compute_partial_derivative(loss1, predictions[0])
    dL_doutput2 = compute_partial_derivative(loss2, predictions[1])
    dL_doutput3 = compute_partial_derivative(loss3, predictions[2])
    
    return [dL_doutput1, dL_doutput2, dL_doutput3]`,
          realWorld: 'Multi-task learning, ensemble models, structured prediction',
          visual: 'Shows multiple output heads with their respective partial derivatives'
        }
      ],
      'gradients': [
        {
          title: 'Gradient Descent Algorithm',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Gradient points to steepest ascent, we move opposite direction',
          stepByStep: [
            'Calculate gradient ∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂wₙ]',
            'Gradient points toward higher loss (bad direction)',
            'Move opposite: w = w - α∇L (toward lower loss)',
            'Repeat until convergence (gradient ≈ 0)'
          ],
          example: 'For 2D: gradient = [∂L/∂x, ∂L/∂y], move in -gradient direction',
          code: `# Gradient descent
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    theta = np.zeros(X.shape[1])
    for i in range(iterations):
        gradient = compute_gradient(theta, X, y)
        theta = theta - learning_rate * gradient
        if np.linalg.norm(gradient) < 1e-6:
            break  # Converged
    return theta`,
          realWorld: 'Training neural networks, logistic regression, all optimization',
          visual: 'Shows path descending a 3D loss surface toward minimum'
        },
        {
          title: 'Stochastic Gradient Descent (SGD)',
          icon: <Zap className="w-6 h-6" />,
          description: 'Use gradient of single sample instead of full dataset',
          stepByStep: [
            'Randomly select one sample (x_i, y_i)',
            'Calculate gradient using only this sample: ∇L_i',
            'Update weights: w = w - α∇L_i',
            'Repeat for all samples (one epoch)',
            'Much faster than full gradient descent'
          ],
          example: 'Process 1 sample at a time instead of all 10,000',
          code: `# Stochastic gradient descent
for epoch in range(num_epochs):
    np.random.shuffle(data)
    for x, y in data:
        gradient = compute_gradient_single_sample(theta, x, y)
        theta = theta - learning_rate * gradient`,
          realWorld: 'Large datasets, online learning, deep learning training',
          visual: 'Shows noisy but faster convergence path'
        },
        {
          title: 'Mini-Batch Gradient Descent',
          icon: <Database className="w-6 h-6" />,
          description: 'Compute gradient using small batches for balance between speed and accuracy',
          stepByStep: [
            'Divide dataset into mini-batches (e.g., 32 samples)',
            'Compute gradient using one mini-batch: ∇L_batch',
            'Update weights: w = w - α∇L_batch',
            'Process all mini-batches (one epoch)',
            'Balances speed (SGD) and accuracy (full gradient)'
          ],
          example: 'Process 32 samples at a time instead of 1 (SGD) or 10,000 (full gradient)',
          code: `# Mini-batch gradient descent
batch_size = 32
for epoch in range(num_epochs):
    for batch in create_batches(data, batch_size):
        gradient = compute_gradient(theta, batch.X, batch.y)
        theta = theta - learning_rate * gradient`,
          realWorld: 'Standard training method for all deep learning - CNNs, RNNs, Transformers',
          visual: 'Shows smoother convergence than SGD, faster than full gradient descent'
        },
        {
          title: 'Gradient Clipping',
          icon: <Shield className="w-6 h-6" />,
          description: 'Prevent exploding gradients by clipping gradient magnitude',
          stepByStep: [
            'Compute gradient: ∇L',
            'Check gradient magnitude: ||∇L||',
            'If ||∇L|| > threshold, clip: ∇L = threshold × ∇L / ||∇L||',
            'Prevents large weight updates',
            'Stabilizes training, especially in RNNs'
          ],
          example: 'If gradient magnitude is 100 and threshold is 10, clip to magnitude 10',
          code: `# Gradient clipping
gradient = compute_gradient(theta, X, y)
gradient_norm = np.linalg.norm(gradient)
max_norm = 10.0

if gradient_norm > max_norm:
    gradient = gradient * (max_norm / gradient_norm)  # Clip

theta = theta - learning_rate * gradient`,
          realWorld: 'RNN/LSTM training, transformer training, preventing training instability',
          visual: 'Shows gradient magnitude before/after clipping, preventing explosions'
        },
        {
          title: 'Second-Order Optimization',
          icon: <Activity className="w-6 h-6" />,
          description: 'Use second derivatives (Hessian) for faster convergence',
          stepByStep: [
            'Compute gradient: ∇L (first derivatives)',
            'Compute Hessian: H (second derivatives)',
            'Hessian captures curvature of loss surface',
            'Newton\'s method: θ = θ - H⁻¹∇L',
            'Converges faster but more expensive to compute'
          ],
          example: 'Hessian tells us not just direction (gradient) but also curvature',
          code: `# Newton's method (second-order)
gradient = compute_gradient(theta, X, y)
hessian = compute_hessian(theta, X, y)  # Second derivatives
# Update: θ = θ - H⁻¹∇L
theta = theta - np.linalg.inv(hessian) @ gradient`,
          realWorld: 'Quasi-Newton methods (L-BFGS), natural gradient descent, advanced optimizers',
          visual: 'Shows faster convergence using curvature information'
        },
        {
          title: 'Gradient-Based Feature Engineering',
          icon: <Settings className="w-6 h-6" />,
          description: 'Use gradients to create new features that improve model performance',
          stepByStep: [
            'Train initial model, compute gradients',
            'Identify features with large gradient magnitudes',
            'Create interactions/transformations of important features',
            'Retrain with new features',
            'Iteratively improve feature set'
          ],
          example: 'If ∂L/∂age and ∂L/∂income are large, create feature age×income',
          code: `# Feature engineering using gradients
initial_model = train_model(X, y)
gradients = compute_feature_gradients(initial_model, X, y)

# Create new features based on gradient importance
important_features = [f for f, g in zip(features, gradients) if abs(g) > threshold]
X_new = create_interactions(X, important_features)`,
          realWorld: 'Automated feature engineering, feature selection, model improvement',
          visual: 'Shows feature importance and new features created from gradients'
        }
      ],
      'chain-rule': [
        {
          title: 'Deep Neural Network Training',
          icon: <Network className="w-6 h-6" />,
          description: 'Chain rule enables gradient computation through multiple layers',
          stepByStep: [
            'Network: input → layer1 → layer2 → ... → output',
            'Forward pass: compute activations layer by layer',
            'Backward pass: apply chain rule layer by layer',
            '∂L/∂w_layer1 = (∂L/∂output) × (∂output/∂layer2) × ... × (∂layer1/∂w_layer1)',
            'Multiply derivatives through the chain'
          ],
          example: '3-layer network: ∂L/∂w₁ = (∂L/∂y) × (∂y/∂h₂) × (∂h₂/∂h₁) × (∂h₁/∂w₁)',
          code: `# Chain rule in deep network
def backward_pass(loss, activations):
    # Start from output
    grad_output = compute_loss_gradient(loss)
    
    # Apply chain rule backward
    grad_layer2 = grad_output * activation_derivative(layer2)
    grad_layer1 = grad_layer2 * activation_derivative(layer1)
    grad_weights = grad_layer1 * input
    
    return grad_weights  # Chain rule multiplies all derivatives`,
          realWorld: 'Training all deep networks - ResNet, VGG, DenseNet, any multi-layer network',
          visual: 'Shows gradient propagation backward through network layers using chain rule'
        },
        {
          title: 'Recurrent Neural Networks (RNNs)',
          icon: <MessageSquare className="w-6 h-6" />,
          description: 'Chain rule handles time-dependent sequences in RNNs',
          stepByStep: [
            'RNN processes sequence: x₁ → x₂ → ... → xₜ',
            'Each time step depends on previous: hₜ = f(hₜ₋₁, xₜ)',
            'Chain rule: ∂L/∂h₁ = (∂L/∂hₜ) × (∂hₜ/∂hₜ₋₁) × ... × (∂h₂/∂h₁)',
            'Backpropagation Through Time (BPTT)',
            'Gradients multiply across time steps'
          ],
          example: 'For sequence of length 10, chain rule multiplies 10 derivatives',
          code: `# Backpropagation Through Time (BPTT)
def bptt(rnn, sequence, loss):
    # Forward pass: store all hidden states
    hidden_states = forward_pass(rnn, sequence)
    
    # Backward pass: chain rule across time
    grad_hidden = compute_loss_gradient(loss)
    for t in reversed(range(len(sequence))):
        grad_hidden = grad_hidden * rnn.activation_derivative(hidden_states[t])
        # Chain rule: multiply by derivative at each time step
    
    return grad_hidden`,
          realWorld: 'Language modeling, speech recognition, time series prediction, LSTM/GRU training',
          visual: 'Shows gradient flow backward through time steps in RNN'
        },
        {
          title: 'Residual Connections (ResNet)',
          icon: <Layers className="w-6 h-6" />,
          description: 'Chain rule handles skip connections in residual networks',
          stepByStep: [
            'Residual block: output = F(x) + x (skip connection)',
            'Gradient has two paths: through F(x) and through x',
            'Chain rule: ∂L/∂x = (∂L/∂output) × (∂output/∂F) × (∂F/∂x) + (∂L/∂output) × 1',
            'Skip connection provides direct gradient path',
            'Prevents vanishing gradients in deep networks'
          ],
          example: 'Gradient flows both through transformation F(x) and skip connection x',
          code: `# ResNet gradient computation
def resnet_backward(grad_output, residual, transformed):
    # Gradient through transformation path
    grad_transform = grad_output * transformation_derivative(transformed)
    
    # Gradient through skip connection (identity, derivative = 1)
    grad_skip = grad_output * 1
    
    # Chain rule: sum both paths
    grad_input = grad_transform + grad_skip
    return grad_input`,
          realWorld: 'Training very deep CNNs (ResNet-152), preventing vanishing gradients',
          visual: 'Shows gradient flow through both transformation and skip connection paths'
        },
        {
          title: 'Attention Mechanisms',
          icon: <Brain className="w-6 h-6" />,
          description: 'Chain rule computes gradients through attention weights',
          stepByStep: [
            'Attention: output = Σ(attention_weights × values)',
            'Attention weights depend on query, key, value',
            'Chain rule: ∂L/∂query = (∂L/∂output) × (∂output/∂attention) × (∂attention/∂query)',
            'Gradients flow through attention computation',
            'Enables training of transformer models'
          ],
          example: 'Transformer self-attention: gradients flow through Q, K, V matrices',
          code: `# Attention gradient computation
def attention_backward(grad_output, attention_weights, values, queries, keys):
    # Gradient through attention weights
    grad_attention = grad_output @ values.T
    
    # Gradient through queries (chain rule)
    grad_queries = grad_attention * attention_derivative_wrt_query(queries, keys)
    
    # Gradient through keys
    grad_keys = grad_attention * attention_derivative_wrt_key(queries, keys)
    
    return grad_queries, grad_keys  # Chain rule applied`,
          realWorld: 'Training Transformers, BERT, GPT models, vision transformers',
          visual: 'Shows gradient flow through attention mechanism using chain rule'
        },
        {
          title: 'Convolutional Neural Networks',
          icon: <Image className="w-6 h-6" />,
          description: 'Chain rule handles convolution operations and pooling layers',
          stepByStep: [
            'CNN: Conv → ReLU → Pool → Conv → ... → Output',
            'Each layer is a function: output = layer(input)',
            'Chain rule: ∂L/∂conv1 = (∂L/∂output) × ... × (∂pool/∂conv1)',
            'Gradients propagate through convolution kernels',
            'Enables training of image recognition models'
          ],
          example: 'VGG-16: chain rule multiplies derivatives through 16 convolutional layers',
          code: `# CNN backpropagation with chain rule
def cnn_backward(grad_output, conv_layers, activations):
    grad = grad_output
    
    # Backward through each layer (chain rule)
    for layer in reversed(conv_layers):
        # Chain rule: multiply by layer derivative
        grad = grad * layer.activation_derivative(activations[layer])
        grad = layer.backward(grad)  # Convolution backward
    
    return grad  # Chain rule applied through all layers`,
          realWorld: 'Training CNNs for image classification, object detection, semantic segmentation',
          visual: 'Shows gradient flow backward through convolutional layers'
        }
      ],
      'backpropagation': [
        {
          title: 'Neural Network Training',
          icon: <Network className="w-6 h-6" />,
          description: 'Chain rule enables gradient computation through layers',
          stepByStep: [
            'Forward pass: Compute output for given input',
            'Calculate loss: Compare output to target',
            'Backward pass: Start from output layer',
            'Use chain rule: ∂L/∂w = (∂L/∂output) × (∂output/∂z) × (∂z/∂w)',
            'Propagate gradients backward layer by layer',
            'Update weights: w = w - α × ∂L/∂w'
          ],
          example: '3-layer network: Output → Hidden2 → Hidden1 → Input',
          code: `# Backpropagation in PyTorch
output = model(input)
loss = criterion(output, target)
loss.backward()  # Computes gradients using chain rule
optimizer.step()  # Updates weights: w = w - lr * grad`,
          realWorld: 'Training all neural networks - CNNs, RNNs, Transformers',
          visual: 'Shows gradient flow backward through network layers'
        },
        {
          title: 'Deep Learning Frameworks',
          icon: <Brain className="w-6 h-6" />,
          description: 'Automatic differentiation computes gradients automatically',
          stepByStep: [
            'Define computation graph (operations)',
            'Forward pass: Execute operations, store intermediate values',
            'Backward pass: Apply chain rule automatically',
            'Framework handles all gradient calculations',
            'No manual derivative computation needed'
          ],
          example: 'PyTorch/TensorFlow compute gradients automatically',
          code: `# Automatic differentiation
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()  # Computes dy/dx automatically
print(x.grad)  # Gradient: 2*x + 3 = 7`,
          realWorld: 'All modern deep learning - PyTorch, TensorFlow, JAX',
          visual: 'Shows computation graph with forward/backward passes'
        },
        {
          title: 'Transfer Learning & Fine-Tuning',
          icon: <Rocket className="w-6 h-6" />,
          description: 'Backpropagation fine-tunes pre-trained models for new tasks',
          stepByStep: [
            'Start with pre-trained model (e.g., ImageNet weights)',
            'Freeze early layers, unfreeze last layers',
            'Backpropagation only updates unfrozen layers',
            'Gradients flow backward but only update selected weights',
            'Much faster than training from scratch'
          ],
          example: 'Fine-tune ResNet-50: freeze first 40 layers, train last 10 layers',
          code: `# Transfer learning with backpropagation
pretrained_model = models.resnet50(pretrained=True)

# Freeze early layers
for param in pretrained_model.layer1.parameters():
    param.requires_grad = False

# Unfreeze last layers
for param in pretrained_model.fc.parameters():
    param.requires_grad = True

# Backpropagation only updates unfrozen layers
loss.backward()  # Gradients computed but only unfrozen weights update`,
          realWorld: 'Medical imaging, custom classifiers, domain adaptation, few-shot learning',
          visual: 'Shows gradient flow with some layers frozen (no updates)'
        },
        {
          title: 'Gradient Checkpointing',
          icon: <Cpu className="w-6 h-6" />,
          description: 'Trade computation for memory by recomputing activations during backprop',
          stepByStep: [
            'Forward pass: store activations at checkpoints only',
            'Backward pass: recompute activations between checkpoints',
            'Reduces memory usage significantly',
            'Enables training larger models on same hardware',
            'Backpropagation still works correctly'
          ],
          example: 'Store activations every 5 layers instead of every layer - 5× memory savings',
          code: `# Gradient checkpointing
def checkpointed_forward(model, x):
    # Store activations at checkpoints
    checkpoints = []
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i % checkpoint_interval == 0:
            checkpoints.append(x)
    return x, checkpoints

# Backward: recompute between checkpoints
def checkpointed_backward(model, grad_output, checkpoints):
    # Recompute activations during backward pass
    # Backpropagation still works via chain rule`,
          realWorld: 'Training large models on limited GPU memory, GPT models, large CNNs',
          visual: 'Shows memory usage comparison with/without checkpointing'
        },
        {
          title: 'Distributed Training',
          icon: <Network className="w-6 h-6" />,
          description: 'Backpropagation works across multiple GPUs/machines',
          stepByStep: [
            'Split batch across multiple devices',
            'Each device computes gradients for its batch',
            'Backpropagation runs on each device independently',
            'Average gradients across all devices',
            'Update weights using averaged gradients'
          ],
          example: '4 GPUs: each computes gradients for 32 samples, average 4 gradients',
          code: `# Distributed backpropagation
def distributed_backward(model, batches):
    gradients = []
    for batch in batches:  # Each on different GPU
        loss = model(batch)
        loss.backward()  # Backpropagation on this GPU
        gradients.append(model.get_gradients())
    
    # Average gradients across all GPUs
    avg_gradient = average(gradients)
    model.update_weights(avg_gradient)`,
          realWorld: 'Training large models (GPT-3, BERT), multi-GPU training, cloud training',
          visual: 'Shows gradient computation across multiple devices, then averaging'
        },
        {
          title: 'Meta-Learning & Few-Shot Learning',
          icon: <Brain className="w-6 h-6" />,
          description: 'Backpropagation through optimization process for meta-learning',
          stepByStep: [
            'Train model on task A using backpropagation',
            'Evaluate on task B',
            'Backpropagate through the training process itself',
            'Update meta-parameters to improve learning',
            'Model learns to learn faster'
          ],
          example: 'MAML: backpropagate through gradient descent steps to learn good initialization',
          code: `# Meta-learning with backpropagation
def meta_learn(model, tasks):
    meta_grad = 0
    for task in tasks:
        # Inner loop: train on task
        for step in range(inner_steps):
            loss = model(task)
            grad = compute_gradient(loss)
            model = model - lr * grad
        
        # Outer loop: backpropagate through training
        meta_loss = evaluate(model, task)
        meta_grad += compute_gradient(meta_loss)
    
    # Update meta-parameters
    meta_params = meta_params - meta_lr * meta_grad`,
          realWorld: 'Few-shot learning, rapid adaptation, learning to learn, MAML, Reptile',
          visual: 'Shows nested optimization with backpropagation through training'
        }
      ]
    },
    'probability-statistics': {
      'descriptive': [
        {
          title: 'Data Preprocessing',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Normalize features using mean and standard deviation',
          stepByStep: [
            'Calculate mean μ and standard deviation σ for each feature',
            'Standardize: z = (x - μ) / σ',
            'All features now have mean=0, std=1',
            'Prevents features with large values from dominating',
            'Speeds up training convergence'
          ],
          example: 'Age: 0-100, Income: 0-1,000,000 → Both become -3 to +3',
          code: `# Feature standardization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std

# Or using sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`,
          realWorld: 'All ML pipelines - preprocessing step before training',
          visual: 'Shows distribution before/after normalization'
        },
        {
          title: 'Outlier Detection',
          icon: <Target className="w-6 h-6" />,
          description: 'Identify unusual data points using z-scores',
          stepByStep: [
            'Calculate mean and standard deviation',
            'Compute z-score: z = (x - μ) / σ',
            'Flag outliers: |z| > 3 (3 standard deviations)',
            'Remove or handle outliers appropriately',
            'Prevents model from learning from bad data'
          ],
          example: 'Customer with $10M income when mean is $50K',
          code: `# Outlier detection using z-score
z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
outliers = np.where(z_scores > 3)[0]
X_clean = np.delete(X, outliers, axis=0)`,
          realWorld: 'Fraud detection, data cleaning, quality assurance',
          visual: 'Shows normal distribution with outliers highlighted'
        },
        {
          title: 'Feature Scaling & Normalization',
          icon: <Settings className="w-6 h-6" />,
          description: 'Use mean and std to scale features for optimal model performance',
          stepByStep: [
            'Calculate mean and standard deviation for each feature',
            'Apply Min-Max scaling: (x - min) / (max - min)',
            'Or Z-score normalization: (x - μ) / σ',
            'All features now on same scale',
            'Prevents features with large ranges from dominating'
          ],
          example: 'Height (cm) and Weight (kg) scaled to [0,1] or mean=0, std=1',
          code: `# Feature scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max scaling (0 to 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Z-score normalization (mean=0, std=1)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)`,
          realWorld: 'Neural networks, SVM, k-means clustering - all benefit from scaled features',
          visual: 'Shows feature distributions before/after scaling'
        },
        {
          title: 'Data Quality Assessment',
          icon: <Shield className="w-6 h-6" />,
          description: 'Use descriptive statistics to assess data quality before modeling',
          stepByStep: [
            'Calculate mean, median, std for each feature',
            'Check for missing values (NaN)',
            'Identify features with zero variance (std = 0)',
            'Detect skewed distributions (mean ≠ median)',
            'Flag potential data quality issues'
          ],
          example: 'If std = 0, feature has no variation - remove it. If mean >> median, data is skewed',
          code: `# Data quality assessment
def assess_data_quality(X):
    stats = {}
    for i, feature in enumerate(X.columns):
        mean = X[feature].mean()
        median = X[feature].median()
        std = X[feature].std()
        
        stats[feature] = {
            'mean': mean,
            'median': median,
            'std': std,
            'zero_variance': std == 0,
            'skewed': abs(mean - median) > 2 * std
        }
    return stats`,
          realWorld: 'Data validation, EDA (Exploratory Data Analysis), preprocessing pipelines',
          visual: 'Shows statistics dashboard with quality flags'
        },
        {
          title: 'Baseline Model Performance',
          icon: <Gauge className="w-6 h-6" />,
          description: 'Use mean/median as simple baseline predictions',
          stepByStep: [
            'Calculate mean or median of target variable',
            'Predict mean/median for all samples',
            'Compare ML model performance against baseline',
            'Model must beat baseline to be useful',
            'Baseline provides performance floor'
          ],
          example: 'If mean house price = $300K, baseline predicts $300K for all houses. Model must do better!',
          code: `# Baseline model
mean_baseline = np.mean(y_train)
baseline_predictions = np.full(len(y_test), mean_baseline)
baseline_mse = np.mean((baseline_predictions - y_test)**2)

# Compare with ML model
model_mse = np.mean((model_predictions - y_test)**2)
improvement = (baseline_mse - model_mse) / baseline_mse`,
          realWorld: 'Model evaluation, performance benchmarking, sanity checks',
          visual: 'Shows baseline predictions vs model predictions'
        }
      ],
      'covariance': [
        {
          title: 'Principal Component Analysis (PCA)',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Use covariance matrix to find principal components for dimensionality reduction',
          stepByStep: [
            'Compute covariance matrix Σ from data',
            'Find eigenvalues and eigenvectors of Σ',
            'Eigenvectors are principal components (directions of max variance)',
            'Project data onto top k principal components',
            'Reduce dimensions while preserving most variance'
          ],
          example: '1000 features → 50 principal components, preserving 95% variance',
          code: `# PCA using covariance matrix
from sklearn.decomposition import PCA

# Compute covariance matrix
cov_matrix = np.cov(X.T)

# PCA (uses covariance internally)
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# Explained variance
variance_ratio = pca.explained_variance_ratio_`,
          realWorld: 'Image compression, feature reduction, visualization, noise reduction',
          visual: 'Shows data projected onto principal components'
        },
        {
          title: 'Feature Selection & Multicollinearity',
          icon: <Target className="w-6 h-6" />,
          description: 'Remove highly correlated features to prevent multicollinearity',
          stepByStep: [
            'Compute correlation matrix for all feature pairs',
            'Identify pairs with |correlation| > threshold (e.g., 0.95)',
            'Remove one feature from each highly correlated pair',
            'Prevents numerical instability in linear models',
            'Reduces model complexity'
          ],
          example: 'If "height_cm" and "height_inches" have correlation = 1.0, remove one',
          code: `# Remove highly correlated features
import numpy as np
import pandas as pd

# Compute correlation matrix
corr_matrix = X.corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

# Remove one feature from each pair
features_to_remove = [pair[1] for pair in high_corr_pairs]
X_clean = X.drop(columns=features_to_remove)`,
          realWorld: 'Linear regression, logistic regression, preventing overfitting',
          visual: 'Shows correlation heatmap with highly correlated pairs highlighted'
        },
        {
          title: 'Multivariate Gaussian Distribution',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Covariance matrix defines multivariate normal distribution',
          stepByStep: [
            'Multivariate Gaussian: N(μ, Σ) where μ is mean vector, Σ is covariance matrix',
            'Covariance matrix captures relationships between variables',
            'Diagonal elements = variances, off-diagonal = covariances',
            'Used in Gaussian Mixture Models, anomaly detection',
            'Enables modeling of correlated features'
          ],
          example: '2D Gaussian: covariance matrix captures how x and y vary together',
          code: `# Multivariate Gaussian
from scipy.stats import multivariate_normal

# Mean vector
mu = np.array([0, 0])

# Covariance matrix
cov = np.array([[1, 0.5], [0.5, 1]])

# Create distribution
dist = multivariate_normal(mu, cov)

# Sample from distribution
samples = dist.rvs(size=1000)`,
          realWorld: 'Gaussian Mixture Models, anomaly detection, density estimation',
          visual: 'Shows 2D/3D Gaussian distribution with covariance ellipses'
        },
        {
          title: 'Feature Engineering & Interactions',
          icon: <Settings className="w-6 h-6" />,
          description: 'Use correlation to identify features for interaction terms',
          stepByStep: [
            'Compute correlation between features',
            'Identify moderately correlated features (0.3-0.7)',
            'Create interaction features: feature1 × feature2',
            'Interaction captures combined effect',
            'Improves model performance'
          ],
          example: 'If age and income are correlated, create "age × income" feature',
          code: `# Feature interactions based on correlation
corr_matrix = X.corr()

# Find moderately correlated pairs
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr = abs(corr_matrix.iloc[i, j])
        if 0.3 < corr < 0.7:
            # Create interaction feature
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            X[f'{feature1}_x_{feature2}'] = X[feature1] * X[feature2]`,
          realWorld: 'Linear models, tree-based models, improving predictive power',
          visual: 'Shows correlation matrix and new interaction features'
        }
      ],
      'conditional-probability': [
        {
          title: 'Decision Tree Splitting',
          icon: <Network className="w-6 h-6" />,
          description: 'Use conditional probabilities to determine optimal splits',
          stepByStep: [
            'At each node, calculate P(class|feature) for each feature',
            'Choose feature that best separates classes',
            'Split maximizes information gain (reduces entropy)',
            'Recursively build tree using conditional probabilities',
            'Leaf nodes predict class with highest probability'
          ],
          example: 'If P(spam|contains "free") = 0.9, split on "contains free" feature',
          code: `# Decision tree using conditional probabilities
from sklearn.tree import DecisionTreeClassifier

# Tree uses conditional probabilities internally
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# At each split, tree calculates:
# P(class|feature_value) for each feature and value
# Chooses split that maximizes information gain`,
          realWorld: 'Classification trees, random forests, gradient boosting (XGBoost, LightGBM)',
          visual: 'Shows tree structure with conditional probabilities at each node'
        },
        {
          title: 'Bayesian Networks',
          icon: <Brain className="w-6 h-6" />,
          description: 'Model conditional dependencies between variables',
          stepByStep: [
            'Define network structure: nodes = variables, edges = dependencies',
            'Each node has conditional probability: P(node|parents)',
            'Chain rule: P(x₁, x₂, ..., xₙ) = Π P(xᵢ|parents(xᵢ))',
            'Inference: compute P(query|evidence)',
            'Learning: estimate conditional probabilities from data'
          ],
          example: 'Medical diagnosis: P(disease|symptoms, age, test_results)',
          code: `# Bayesian Network
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Define structure
model = BayesianModel([('Age', 'Disease'), ('Symptoms', 'Disease'), ('Disease', 'Test')])

# Learn conditional probabilities
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
prob = infer.query(['Disease'], evidence={'Symptoms': 'fever', 'Age': 50})`,
          realWorld: 'Medical diagnosis, risk assessment, probabilistic reasoning systems',
          visual: 'Shows network graph with conditional probability tables'
        },
        {
          title: 'Hidden Markov Models (HMMs)',
          icon: <Activity className="w-6 h-6" />,
          description: 'Model sequences using conditional probabilities',
          stepByStep: [
            'HMM has hidden states and observations',
            'Transition probability: P(state_t|state_{t-1})',
            'Emission probability: P(observation_t|state_t)',
            'Use chain rule to compute P(sequence)',
            'Viterbi algorithm finds most likely state sequence'
          ],
          example: 'Speech recognition: P(phoneme|audio_features) and P(phoneme_t|phoneme_{t-1})',
          code: `# Hidden Markov Model
from hmmlearn import hmm

# Create HMM
model = hmm.GaussianHMM(n_components=3)

# Train (learns transition and emission probabilities)
model.fit(observations)

# Decode (find most likely state sequence)
states = model.decode(observations)[1]

# Transition: P(state_t|state_{t-1})
# Emission: P(obs_t|state_t)`,
          realWorld: 'Speech recognition, natural language processing, bioinformatics, time series',
          visual: 'Shows state transitions and emission probabilities'
        },
        {
          title: 'Conditional Random Fields (CRFs)',
          icon: <Layers className="w-6 h-6" />,
          description: 'Model conditional probability of label sequence given observations',
          stepByStep: [
            'CRF models P(label_sequence|observation_sequence)',
            'Uses conditional probabilities, not joint probabilities',
            'Considers context: P(label_t|observations, label_{t-1})',
            'More flexible than HMMs (no independence assumptions)',
            'Used for sequence labeling tasks'
          ],
          example: 'Named entity recognition: P(entity_tags|words)',
          code: `# Conditional Random Field
from sklearn_crfsuite import CRF

# Create CRF model
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1)

# Train (learns conditional probabilities)
crf.fit(X_train, y_train)

# Predict sequences
predictions = crf.predict(X_test)

# Models P(y|x) directly, not P(x,y)`,
          realWorld: 'Named entity recognition, part-of-speech tagging, information extraction',
          visual: 'Shows label sequence with conditional dependencies'
        }
      ],
      'bayes': [
        {
          title: 'Naive Bayes Classifier',
          icon: <Brain className="w-6 h-6" />,
          description: 'Use Bayes theorem for classification',
          stepByStep: [
            'Calculate prior: P(class) from training data',
            'Calculate likelihood: P(feature|class) for each feature',
            'Assume feature independence: P(features|class) = Π P(feature_i|class)',
            'Apply Bayes: P(class|features) = P(features|class) × P(class) / P(features)',
            'Predict class with highest probability'
          ],
          example: 'Spam detection: P(spam|words) = P(words|spam) × P(spam) / P(words)',
          code: `# Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Manual calculation
P_spam = count_spam / total_emails
P_words_given_spam = product(P(word_i | spam) for each word)
P_spam_given_words = P_words_given_spam * P_spam / P_words`,
          realWorld: 'Spam filtering, text classification, medical diagnosis',
          visual: 'Shows probability calculation tree'
        },
        {
          title: 'Bayesian Inference',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Update beliefs with new evidence',
          stepByStep: [
            'Start with prior belief: P(hypothesis)',
            'Observe data: Calculate likelihood P(data|hypothesis)',
            'Apply Bayes theorem: P(hypothesis|data) = P(data|hypothesis) × P(hypothesis) / P(data)',
            'Posterior becomes new prior for next observation',
            'Beliefs converge to true value with more data'
          ],
          example: 'Medical test: Update disease probability after test result',
          code: `# Bayesian inference
prior = 0.01  # 1% disease prevalence
likelihood = 0.95  # Test accuracy
evidence = prior * likelihood + (1 - prior) * (1 - likelihood)
posterior = (likelihood * prior) / evidence`,
          realWorld: 'A/B testing, medical diagnosis, recommendation systems',
          visual: 'Shows prior → likelihood → posterior update'
        },
        {
          title: 'Bayesian Optimization',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Use Bayes theorem to optimize hyperparameters efficiently',
          stepByStep: [
            'Start with prior over hyperparameter space',
            'Evaluate objective function at sample points',
            'Update posterior using Bayes theorem',
            'Use acquisition function (e.g., Expected Improvement)',
            'Select next point to evaluate, repeat'
          ],
          example: 'Optimize learning rate: start with prior, evaluate model performance, update beliefs',
          code: `# Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real

# Define search space
space = [Real(0.001, 0.1, name='learning_rate')]

# Objective function
def objective(params):
    lr = params[0]
    model = train_model(learning_rate=lr)
    return -model.score(X_test, y_test)  # Minimize negative score

# Optimize using Bayesian approach
result = gp_minimize(objective, space, n_calls=50)`,
          realWorld: 'Hyperparameter tuning, neural architecture search, AutoML',
          visual: 'Shows hyperparameter space with posterior distribution and acquisition function'
        },
        {
          title: 'Bayesian Neural Networks',
          icon: <Network className="w-6 h-6" />,
          description: 'Use Bayesian inference to quantify uncertainty in neural networks',
          stepByStep: [
            'Place prior distributions on network weights',
            'Use Bayes theorem to compute posterior P(weights|data)',
            'Posterior captures uncertainty in weights',
            'Predictions become distributions, not point estimates',
            'Provides uncertainty quantification'
          ],
          example: 'Instead of fixed weights, have weight distributions - captures model uncertainty',
          code: `# Bayesian Neural Network
import tensorflow_probability as tfp

# Define prior on weights
def prior(kernel_size, bias_size, dtype=None):
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros(kernel_size + bias_size), scale=1.0),
        reinterpreted_batch_ndims=1)

# Define likelihood
def likelihood(dist):
    return tfp.distributions.Normal(loc=dist, scale=1.0)

# Build Bayesian layer
layer = tfp.layers.DenseVariational(
    units=1,
    make_prior_fn=prior,
    make_posterior_fn=posterior,
    kl_weight=1/X_train.shape[0],
    activation='sigmoid')`,
          realWorld: 'Uncertainty quantification, active learning, risk-sensitive applications',
          visual: 'Shows weight distributions and prediction uncertainty'
        },
        {
          title: 'Recommendation Systems',
          icon: <Users className="w-6 h-6" />,
          description: 'Use Bayesian inference to update user preferences',
          stepByStep: [
            'Start with prior on user preferences',
            'Observe user interactions (likes, clicks, purchases)',
            'Update posterior: P(preferences|interactions)',
            'Recommend items with high posterior probability',
            'Continually update as more data arrives'
          ],
          example: 'Netflix: Start with general preferences, update based on viewing history',
          code: `# Bayesian Recommendation System
# Prior: P(user_likes_genre)
prior = {'action': 0.3, 'comedy': 0.3, 'drama': 0.4}

# Likelihood: P(interaction|user_likes_genre)
likelihood = compute_likelihood(user_interactions, genres)

# Posterior: P(user_likes_genre|interactions)
posterior = {}
for genre in prior:
    posterior[genre] = (likelihood[genre] * prior[genre]) / evidence

# Recommend based on posterior
recommendations = sorted(posterior.items(), key=lambda x: x[1], reverse=True)`,
          realWorld: 'Netflix, Amazon, Spotify, e-commerce personalization',
          visual: 'Shows preference updates from prior to posterior'
        }
      ],
      'distributions': [
        {
          title: 'Generative Models',
          icon: <Image className="w-6 h-6" />,
          description: 'Model data distribution to generate new samples',
          stepByStep: [
            'Learn probability distribution P(x) from training data',
            'Sample from distribution to generate new data',
            'Normal distribution for continuous data',
            'Bernoulli/Binomial for binary/categorical data',
            'Generate realistic new samples'
          ],
          example: 'GANs generate images by learning image distribution',
          code: `# Generative model
from scipy.stats import norm

# Learn distribution parameters
mu, sigma = np.mean(data), np.std(data)

# Generate new samples
new_samples = norm.rvs(mu, sigma, size=1000)

# Or using neural networks (GANs)
generator = Generator()
fake_images = generator(noise)`,
          realWorld: 'Image generation (DALL-E, Midjourney), data augmentation',
          visual: 'Shows learned distribution and generated samples'
        },
        {
          title: 'Loss Function Design',
          icon: <Target className="w-6 h-6" />,
          description: 'Assume error distribution to design appropriate loss',
          stepByStep: [
            'Assume error follows normal distribution',
            'Maximize likelihood = minimize negative log-likelihood',
            'Leads to Mean Squared Error (MSE) loss',
            'For classification: assume Bernoulli → Cross-entropy loss',
            'Loss function matches data distribution'
          ],
          example: 'Regression: MSE assumes Gaussian errors',
          code: `# Loss functions based on distributions
# MSE (assumes normal distribution)
mse_loss = np.mean((y_pred - y_true)**2)

# Cross-entropy (assumes Bernoulli distribution)
ce_loss = -np.mean(y_true * np.log(y_pred) + 
                   (1 - y_true) * np.log(1 - y_pred))`,
          realWorld: 'All ML models - choosing right loss for problem type',
          visual: 'Shows error distribution and corresponding loss function'
        },
        {
          title: 'Variational Autoencoders (VAEs)',
          icon: <Image className="w-6 h-6" />,
          description: 'Learn latent distributions to generate diverse outputs',
          stepByStep: [
            'Encode input to latent distribution parameters (μ, σ)',
            'Sample from latent distribution: z ~ N(μ, σ)',
            'Decode z to reconstruct input',
            'Learn to match data distribution',
            'Generate new samples by sampling from learned distribution'
          ],
          example: 'VAE learns distribution of faces, generates new realistic faces',
          code: `# Variational Autoencoder
import torch
import torch.nn as nn

class VAE(nn.Module):
    def encode(self, x):
        # Output distribution parameters
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar`,
          realWorld: 'Image generation, data augmentation, representation learning',
          visual: 'Shows latent distribution and generated samples'
        },
        {
          title: 'Gaussian Mixture Models (GMMs)',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Model data as mixture of multiple Gaussian distributions',
          stepByStep: [
            'Assume data comes from k Gaussian distributions',
            'Each component has its own mean and covariance',
            'Learn component parameters using EM algorithm',
            'Assign data points to components probabilistically',
            'Use for clustering and density estimation'
          ],
          example: 'Customer segmentation: model customer data as mixture of different customer types',
          code: `# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture

# Create GMM with k components
gmm = GaussianMixture(n_components=3)

# Fit model (learns component distributions)
gmm.fit(X)

# Predict component assignments
labels = gmm.predict(X)

# Get component parameters
means = gmm.means_
covariances = gmm.covariances_`,
          realWorld: 'Clustering, anomaly detection, density estimation, customer segmentation',
          visual: 'Shows multiple Gaussian distributions and data assignments'
        },
        {
          title: 'Monte Carlo Methods',
          icon: <Activity className="w-6 h-6" />,
          description: 'Sample from distributions to estimate expectations',
          stepByStep: [
            'Define target distribution P(x)',
            'Sample many points from distribution',
            'Estimate expectation: E[f(x)] ≈ (1/N) Σ f(xᵢ)',
            'More samples = better estimate',
            'Used when exact computation is intractable'
          ],
          example: 'Estimate integral: ∫ f(x)P(x)dx ≈ (1/N) Σ f(xᵢ) where xᵢ ~ P(x)',
          code: `# Monte Carlo estimation
import numpy as np
from scipy.stats import norm

# Sample from distribution
samples = norm.rvs(loc=0, scale=1, size=10000)

# Estimate expectation
def f(x):
    return x**2

expectation = np.mean(f(samples))

# True value: E[X²] = Var(X) + E[X]² = 1 + 0 = 1`,
          realWorld: 'Bayesian inference, reinforcement learning, uncertainty propagation',
          visual: 'Shows samples from distribution and estimated statistics'
        },
        {
          title: 'Confidence Intervals & Prediction Intervals',
          icon: <Gauge className="w-6 h-6" />,
          description: 'Use distributions to quantify prediction uncertainty',
          stepByStep: [
            'Assume prediction errors follow distribution (e.g., Normal)',
            'Estimate distribution parameters from residuals',
            'Compute confidence interval: [μ - z×σ, μ + z×σ]',
            '95% CI: z = 1.96 (for Normal distribution)',
            'Provides uncertainty bounds for predictions'
          ],
          example: 'House price prediction: $300K ± $20K (95% confidence interval)',
          code: `# Prediction intervals
from scipy.stats import norm

# Predictions
y_pred = model.predict(X_test)

# Residuals (errors)
residuals = y_test - y_pred

# Estimate error distribution
error_mean = np.mean(residuals)
error_std = np.std(residuals)

# 95% prediction interval
z = 1.96
lower_bound = y_pred - z * error_std
upper_bound = y_pred + z * error_std`,
          realWorld: 'Risk assessment, decision making, uncertainty quantification',
          visual: 'Shows predictions with confidence intervals'
        }
      ]
    },
    'supervised-learning': {
      'foundations': [
        {
          title: 'Linear Regression: Combining Linear Algebra and Calculus',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Using matrices for data representation and calculus for optimization',
          stepByStep: [
            'Represent dataset as matrix X (m samples × n features)',
            'Represent parameters as vector θ (n weights)',
            'Make predictions: ŷ = Xθ (matrix multiplication)',
            'Calculate loss: J(θ) = (1/2m) × Σ(ŷ - y)²',
            'Compute gradient: ∂J/∂θ = (1/m) × Xᵀ(ŷ - y)',
            'Update parameters: θ := θ - α × ∂J/∂θ',
            'Repeat until convergence'
          ],
          example: 'House price prediction: features (size, bedrooms) → price',
          code: `# Linear Regression using Linear Algebra and Calculus
import numpy as np

# Data representation (Linear Algebra)
X = np.array([[1, x1], [1, x2], ...])  # m × n matrix
y = np.array([y1, y2, ...])            # m × 1 vector
theta = np.zeros(n)                    # n × 1 vector

# Training (Calculus)
for epoch in range(num_epochs):
    # Prediction (Linear Algebra)
    y_pred = X @ theta
    
    # Loss calculation
    loss = (1/(2*m)) * np.sum((y_pred - y)**2)
    
    # Gradient (Calculus)
    gradient = (1/m) * X.T @ (y_pred - y)
    
    # Update (Optimization)
    theta = theta - learning_rate * gradient`,
          realWorld: 'All regression problems: price prediction, demand forecasting, trend analysis',
          visual: 'Shows matrix operations for predictions and gradient descent optimization'
        },
        {
          title: 'Logistic Regression: Probability Meets Linear Algebra',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Combining linear algebra with probability for classification',
          stepByStep: [
            'Use linear algebra: z = θᵀx (linear combination)',
            'Apply sigmoid function: P(y=1|x) = 1/(1+e^(-z))',
            'This gives probability (Probability)',
            'Calculate cross-entropy loss',
            'Use gradient descent (Calculus) to optimize',
            'Make predictions based on probability threshold'
          ],
          example: 'Spam detection: email features → probability of spam',
          code: `# Logistic Regression combining Linear Algebra, Calculus, Probability
import numpy as np

# Linear combination (Linear Algebra)
z = X @ theta

# Probability (Probability & Statistics)
prob = 1 / (1 + np.exp(-z))  # Sigmoid function

# Loss (Probability)
loss = -np.mean(y * np.log(prob) + (1-y) * np.log(1-prob))

# Gradient (Calculus)
gradient = (1/m) * X.T @ (prob - y)

# Update (Optimization)
theta = theta - learning_rate * gradient`,
          realWorld: 'Binary classification: spam detection, medical diagnosis, fraud detection',
          visual: 'Shows how linear algebra transforms to probabilities via sigmoid'
        },
        {
          title: 'Naive Bayes: Pure Probability Approach',
          icon: <Brain className="w-6 h-6" />,
          description: 'Using Bayes theorem and probability distributions for classification',
          stepByStep: [
            'Estimate prior probabilities: P(y) for each class',
            'Estimate likelihoods: P(xᵢ|y) for each feature',
            'Apply Bayes theorem: P(y|x) = P(y) × P(x|y) / P(x)',
            'Use "naive" assumption: features are independent',
            'Calculate: P(y|x₁,...,xₙ) = P(y) × Π P(xᵢ|y)',
            'Predict class with highest probability'
          ],
          example: 'Text classification: word frequencies → document category',
          code: `# Naive Bayes using Probability
from sklearn.naive_bayes import GaussianNB

# Fit model (learns P(y) and P(x|y))
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions (uses Bayes theorem)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

# Under the hood:
# P(y|x) = P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y) / P(x)
# Predict class with max P(y|x)`,
          realWorld: 'Text classification, spam filtering, sentiment analysis, medical diagnosis',
          visual: 'Shows probability distributions and Bayes theorem calculation'
        },
        {
          title: 'Polynomial Regression: Non-Linear Relationships',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Using polynomial features to capture non-linear relationships',
          stepByStep: [
            'Create polynomial features: x, x², x³, ...',
            'Use linear algebra: X_poly = [x, x², x³, ...]',
            'Apply linear regression: ŷ = X_poly × θ',
            'Optimize using calculus (gradient descent)',
            'Higher degree = more complex, risk of overfitting'
          ],
          example: 'House price vs size: Linear might underfit. Polynomial degree 2-3 captures curvature.',
          code: `# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Apply linear regression
model = LinearRegression()
model.fit(X_poly, y)

# Prediction uses polynomial features
y_pred = model.predict(poly.transform(X_test))`,
          realWorld: 'Non-linear regression, curve fitting, relationship modeling',
          visual: 'Shows polynomial curves of different degrees fitting data'
        },
        {
          title: 'Multi-Layer Perceptron: Combining Foundations',
          icon: <Network className="w-6 h-6" />,
          description: 'Neural networks combine all mathematical foundations',
          stepByStep: [
            'Linear Algebra: Each layer = matrix multiplication',
            'Non-linearity: Activation functions (sigmoid, ReLU)',
            'Calculus: Backpropagation uses chain rule for gradients',
            'Probability: Output probabilities for classification',
            'Optimization: Gradient descent updates all weights'
          ],
          example: 'MLP for image classification: Linear algebra (matrix ops) + Calculus (backprop) + Probability (softmax)',
          code: `# Multi-Layer Perceptron
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear Algebra: matrix multiplication layers
        self.fc1 = nn.Linear(784, 256)  # Linear transformation
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Linear Algebra: X @ W + b
        x = nn.functional.relu(self.fc1(x))  # Non-linearity
        x = nn.functional.relu(self.fc2(x))
        # Probability: softmax for classification
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x

# Training uses Calculus (backpropagation) and Optimization (gradient descent)`,
          realWorld: 'All neural network applications: image recognition, NLP, recommendation systems',
          visual: 'Shows how all mathematical foundations combine in neural networks'
        }
      ],
      'key-concepts': [
        {
          title: 'When to Use Regression vs Classification',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Choosing the right problem type based on output requirements',
          stepByStep: [
            'Identify the type of output you need to predict',
            'If output is continuous (numbers): Use Regression',
            'If output is categorical (labels): Use Classification',
            'Consider the business problem: "how much" vs "which category"',
            'Select appropriate loss function and metrics'
          ],
          example: 'House price prediction = Regression (continuous). Spam detection = Classification (spam/not spam)',
          code: `# Regression Example
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  # y_train: continuous values
predictions = model.predict(X_test)  # Returns: [125000.5, 230000.3, ...]

# Classification Example
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)  # y_train: class labels [0, 1, 0, ...]
predictions = model.predict(X_test)  # Returns: [0, 1, 0, 1, ...]`,
          realWorld: 'Every ML problem starts with determining if it\'s regression or classification',
          visual: 'Shows examples of regression (continuous line) vs classification (discrete categories)'
        },
        {
          title: 'Support Vector Machines for Text Classification',
          icon: <Zap className="w-6 h-6" />,
          description: 'Using SVM for high-dimensional text data classification',
          stepByStep: [
            'Convert text to numerical features (TF-IDF, word embeddings)',
            'High-dimensional sparse feature space (thousands of features)',
            'SVM finds optimal hyperplane in this high-dimensional space',
            'Support vectors are the "hard" examples near decision boundary',
            'Kernel trick can handle non-linear relationships in text'
          ],
          example: 'Spam email classification: thousands of word features → SVM finds optimal boundary',
          code: `# SVM for Text Classification
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to features
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(text_train)
X_test = vectorizer.transform(text_test)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

# SVM works well with high-dimensional sparse data like text`,
          realWorld: 'Text classification, spam detection, sentiment analysis, document categorization',
          visual: 'Shows SVM decision boundary in high-dimensional feature space'
        },
        {
          title: 'Multi-class Classification Strategies',
          icon: <Target className="w-6 h-6" />,
          description: 'Handling classification with more than two classes',
          stepByStep: [
            'One-vs-Rest (OvR): Train one classifier per class vs all others',
            'One-vs-One (OvO): Train classifier for each pair of classes',
            'Multinomial: Direct multi-class (e.g., softmax in neural networks)',
            'Choose strategy based on algorithm and problem size',
            'Evaluate using multi-class metrics (confusion matrix, F1-macro)'
          ],
          example: 'Image classification: cat/dog/bird/horse → use multinomial or OvR',
          code: `# Multi-class Classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# One-vs-Rest approach
ovr_classifier = OneVsRestClassifier(SVC(kernel='linear'))
ovr_classifier.fit(X_train, y_train)  # y_train: [0, 1, 2, 3, ...]
predictions = ovr_classifier.predict(X_test)

# Or use multinomial (Logistic Regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)`,
          realWorld: 'Image recognition, document classification, medical diagnosis with multiple diseases',
          visual: 'Shows decision boundaries for multiple classes'
        },
        {
          title: 'Decision Trees for Interpretable Classification',
          icon: <Network className="w-6 h-6" />,
          description: 'Using decision trees for interpretable, rule-based classification',
          stepByStep: [
            'Split data based on feature values',
            'Choose splits that maximize information gain',
            'Recursively build tree until stopping criteria',
            'Leaf nodes predict class',
            'Tree structure provides interpretable rules'
          ],
          example: 'Medical diagnosis: "If age > 50 AND blood_pressure > 140 THEN high_risk"',
          code: `# Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
tree.fit(X_train, y_train)

# Interpretable rules
from sklearn.tree import export_text
rules = export_text(tree, feature_names=feature_names)
print(rules)  # Shows if-then rules`,
          realWorld: 'Medical diagnosis, credit scoring, interpretable ML, rule-based systems',
          visual: 'Shows tree structure with decision rules'
        },
        {
          title: 'K-Nearest Neighbors (KNN) for Instance-Based Learning',
          icon: <Users className="w-6 h-6" />,
          description: 'Using KNN for simple, non-parametric classification/regression',
          stepByStep: [
            'Store all training examples',
            'For new sample, find k nearest neighbors',
            'For classification: majority vote of neighbors',
            'For regression: average of neighbors',
            'No explicit training phase (lazy learning)'
          ],
          example: 'Recommendation: Find 5 users most similar to you, recommend what they liked.',
          code: `# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction finds k nearest neighbors
predictions = knn.predict(X_test)

# For regression
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5)`,
          realWorld: 'Recommendation systems, pattern recognition, simple baseline models',
          visual: 'Shows k nearest neighbors for a query point'
        },
        {
          title: 'Gradient Boosting: Sequential Model Building',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Building strong models by combining weak learners sequentially',
          stepByStep: [
            'Train initial weak model (e.g., shallow tree)',
            'Calculate residuals (errors)',
            'Train next model to predict residuals',
            'Add to ensemble: F(x) = F_prev(x) + α × h(x)',
            'Repeat until convergence'
          ],
          example: 'XGBoost, LightGBM: Sequentially add trees that correct previous errors.',
          code: `# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gbm.fit(X_train, y_train)

# Each tree corrects errors of previous trees`,
          realWorld: 'XGBoost, LightGBM, CatBoost - state-of-the-art for tabular data',
          visual: 'Shows sequential addition of trees correcting errors'
        }
      ],
      'loss-functions': [
        {
          title: 'Choosing the Right Loss Function',
          icon: <Target className="w-6 h-6" />,
          description: 'Different loss functions for different problem types',
          stepByStep: [
            'For regression: Use MSE (sensitive to outliers) or MAE (robust)',
            'For binary classification: Use Binary Cross-Entropy',
            'For multi-class classification: Use Categorical Cross-Entropy',
            'For SVM: Use Hinge Loss',
            'Loss function choice affects model behavior and optimization'
          ],
          example: 'MSE for house price prediction, Cross-entropy for spam detection',
          code: `# Regression: Mean Squared Error
mse = np.mean((y_pred - y_true)**2)

# Binary Classification: Cross-Entropy
ce = -np.mean(y_true * np.log(y_pred) + 
              (1 - y_true) * np.log(1 - y_pred))

# Multi-class: Categorical Cross-Entropy
cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))`,
          realWorld: 'Every ML model requires appropriate loss function - regression vs classification',
          visual: 'Shows how different loss functions penalize errors differently'
        },
        {
          title: 'Loss Function Optimization',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Minimizing loss guides model training',
          stepByStep: [
            'Initialize model parameters randomly',
            'Compute loss on training data',
            'Calculate gradient of loss with respect to parameters',
            'Update parameters: θ = θ - α × ∇L',
            'Repeat until loss converges to minimum'
          ],
          example: 'Gradient descent minimizes MSE to find best linear regression line',
          code: `# Gradient descent optimization
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X)
    loss = loss_function(predictions, y)
    
    # Backward pass
    gradients = compute_gradients(loss, model.parameters())
    
    # Update parameters
    for param, grad in zip(model.parameters(), gradients):
        param = param - learning_rate * grad`,
          realWorld: 'All supervised learning algorithms: neural networks, linear/logistic regression, SVM',
          visual: 'Shows loss decreasing over training iterations'
        },
        {
          title: 'Robust Loss Functions for Outliers',
          icon: <Shield className="w-6 h-6" />,
          description: 'Using robust loss functions when data contains outliers',
          stepByStep: [
            'Identify if data has outliers (check distribution)',
            'For regression with outliers: Use MAE or Huber loss instead of MSE',
            'MAE is less sensitive to outliers than MSE',
            'Huber loss combines benefits of MSE and MAE',
            'Robust loss prevents outliers from dominating training'
          ],
          example: 'House price prediction: A few mansions ($10M) shouldn\'t dominate training. Use MAE instead of MSE.',
          code: `# Robust loss functions
# Mean Absolute Error (robust to outliers)
mae = np.mean(np.abs(y_pred - y_true))

# Huber Loss (combines MSE and MAE)
def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, squared_loss, linear_loss))`,
          realWorld: 'Financial data, sensor data, real-world datasets with anomalies',
          visual: 'Shows how different loss functions handle outliers'
        },
        {
          title: 'Focal Loss for Class Imbalance',
          icon: <Target className="w-6 h-6" />,
          description: 'Using focal loss to handle imbalanced classification problems',
          stepByStep: [
            'Standard cross-entropy treats all samples equally',
            'Focal loss down-weights easy examples',
            'Focuses learning on hard examples',
            'Reduces impact of class imbalance',
            'Improves performance on rare classes'
          ],
          example: 'Medical diagnosis: 99% healthy, 1% disease. Focal loss focuses on rare disease cases.',
          code: `# Focal Loss for class imbalance
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Cross-entropy
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # Focal term: down-weight easy examples
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = (1 - p_t)**gamma
    
    # Alpha weighting for class imbalance
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    return np.mean(alpha_t * focal_weight * ce)`,
          realWorld: 'Object detection, medical diagnosis, fraud detection, rare event prediction',
          visual: 'Shows how focal loss focuses on hard examples'
        },
        {
          title: 'Custom Loss Functions for Business Goals',
          icon: <Settings className="w-6 h-6" />,
          description: 'Designing loss functions that align with business objectives',
          stepByStep: [
            'Identify business objective (e.g., maximize profit, minimize false negatives)',
            'Design loss function that penalizes costly mistakes',
            'Weight different types of errors differently',
            'Train model with custom loss',
            'Evaluate on business metrics, not just accuracy'
          ],
          example: 'Fraud detection: False negative (missed fraud) costs $1000, false positive costs $1. Weight loss accordingly.',
          code: `# Custom loss function for business goals
def business_loss(y_true, y_pred, fn_cost=1000, fp_cost=1):
    # False negatives: predicted negative, actually positive
    fn = np.sum((y_pred == 0) & (y_true == 1)) * fn_cost
    
    # False positives: predicted positive, actually negative
    fp = np.sum((y_pred == 1) & (y_true == 0)) * fp_cost
    
    return fn + fp

# Use in training
# Modify standard loss to incorporate business costs`,
          realWorld: 'Fraud detection, medical diagnosis, recommendation systems, any cost-sensitive application',
          visual: 'Shows how custom loss aligns with business objectives'
        }
      ],
      'model-evaluation': [
        {
          title: 'Classification Model Evaluation',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Using confusion matrix and metrics to assess classifier performance',
          stepByStep: [
            'Split data into train/validation/test sets',
            'Train model on training set',
            'Make predictions on validation/test set',
            'Build confusion matrix: TP, TN, FP, FN',
            'Calculate metrics: Accuracy, Precision, Recall, F1',
            'Plot ROC curve and calculate AUC'
          ],
          example: 'Evaluating spam classifier: high precision = fewer false positives',
          code: `# Calculate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, y_pred)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)`,
          realWorld: 'Evaluating all classification models: spam detection, medical diagnosis, fraud detection',
          visual: 'Shows confusion matrix and ROC curve visualization'
        },
        {
          title: 'Threshold Selection',
          icon: <Target className="w-6 h-6" />,
          description: 'Choosing optimal classification threshold based on use case',
          stepByStep: [
            'Understand business requirements: precision vs recall',
            'For high precision: Use higher threshold (fewer false positives)',
            'For high recall: Use lower threshold (catch more positives)',
            'Plot precision-recall curve',
            'Choose threshold that balances both metrics'
          ],
          example: 'Medical diagnosis: prefer high recall (catch all diseases). Spam filter: prefer high precision (fewer false positives)',
          code: `# Precision-Recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Find optimal threshold (e.g., F1 score)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]`,
          realWorld: 'All binary classification problems require threshold tuning',
          visual: 'Shows how threshold affects precision and recall'
        },
        {
          title: 'Regression Model Evaluation',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Using RMSE, MAE, and R² to evaluate regression models',
          stepByStep: [
            'Split data into train/validation/test sets',
            'Train regression model on training set',
            'Make predictions on test set',
            'Calculate RMSE: sqrt(mean((y_pred - y_true)²))',
            'Calculate MAE: mean(|y_pred - y_true|)',
            'Calculate R²: measures explained variance'
          ],
          example: 'House price prediction: RMSE = $50K means average error is $50K. R² = 0.85 means model explains 85% variance.',
          code: `# Regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# R-squared (coefficient of determination)
r2 = r2_score(y_true, y_pred)

# Interpretation:
# RMSE: Lower is better, in same units as target
# MAE: Lower is better, robust to outliers
# R²: Higher is better, 1.0 = perfect, 0.0 = baseline`,
          realWorld: 'Price prediction, demand forecasting, time series prediction, any regression problem',
          visual: 'Shows predicted vs actual values with error metrics'
        },
        {
          title: 'Multi-Class Classification Metrics',
          icon: <Target className="w-6 h-6" />,
          description: 'Evaluating models with more than two classes',
          stepByStep: [
            'Build confusion matrix for all classes',
            'Calculate per-class precision, recall, F1',
            'Calculate macro-averaged metrics (average across classes)',
            'Calculate micro-averaged metrics (pool all predictions)',
            'Use appropriate averaging based on class imbalance'
          ],
          example: 'Image classification: 10 classes (cat, dog, bird, ...). Calculate metrics for each class and average.',
          code: `# Multi-class metrics
from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Classification report (per-class metrics)
report = classification_report(y_true, y_pred, 
                              target_names=['cat', 'dog', 'bird', ...])

# Macro-averaged F1 (treats all classes equally)
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_true, y_pred, average='macro')

# Micro-averaged F1 (pools all predictions)
micro_f1 = f1_score(y_true, y_pred, average='micro')`,
          realWorld: 'Image classification, document categorization, medical diagnosis with multiple diseases',
          visual: 'Shows multi-class confusion matrix and per-class metrics'
        },
        {
          title: 'Learning Curves for Model Diagnosis',
          icon: <Activity className="w-6 h-6" />,
          description: 'Using learning curves to diagnose bias-variance issues',
          stepByStep: [
            'Train model on increasing training set sizes',
            'Calculate training and validation scores at each size',
            'Plot learning curves (score vs training size)',
            'Analyze gap between curves: large gap = overfitting',
            'Analyze convergence: both curves plateau = need more data or better features'
          ],
          example: 'If validation score plateaus but training score keeps improving → overfitting. If both plateau → underfitting.',
          code: `# Learning curves
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()`,
          realWorld: 'Model diagnosis, determining if more data will help, detecting overfitting/underfitting',
          visual: 'Shows learning curves with training and validation scores'
        }
      ],
      'bias-variance': [
        {
          title: 'Preventing Overfitting',
          icon: <Network className="w-6 h-6" />,
          description: 'Understanding and managing bias-variance tradeoff',
          stepByStep: [
            'Monitor training vs validation loss',
            'If training loss << validation loss: overfitting (high variance)',
            'If both losses high: underfitting (high bias)',
            'Use validation set to find optimal model complexity',
            'Apply regularization or early stopping'
          ],
          example: 'Neural network: too many layers = overfitting, too few = underfitting',
          code: `# Monitor bias-variance
train_loss = model.evaluate(X_train, y_train)
val_loss = model.evaluate(X_val, y_val)

# Overfitting: train_loss << val_loss
# Underfitting: train_loss ≈ val_loss (both high)
# Good fit: train_loss ≈ val_loss (both low)

# Solutions:
# - Reduce model complexity (fewer layers/neurons)
# - Add regularization (L1/L2)
# - Use dropout
# - Early stopping`,
          realWorld: 'All ML models: neural networks, decision trees, polynomial regression',
          visual: 'Shows training vs validation loss curves'
        },
        {
          title: 'Cross-Validation for Model Selection',
          icon: <Layers className="w-6 h-6" />,
          description: 'Using k-fold cross-validation to estimate model performance',
          stepByStep: [
            'Split data into k folds',
            'Train on k-1 folds, validate on 1 fold',
            'Repeat k times, each fold as validation once',
            'Average validation scores across folds',
            'Select model with best average performance'
          ],
          example: '5-fold CV: train on 80% data, validate on 20%, repeat 5 times',
          code: `# K-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_score = scores.mean()
std_score = scores.std()

# Better estimate of model performance
# Reduces variance in performance estimate`,
          realWorld: 'Model selection, hyperparameter tuning, feature selection',
          visual: 'Shows data splits and validation scores across folds'
        },
        {
          title: 'Ensemble Methods to Reduce Variance',
          icon: <Layers className="w-6 h-6" />,
          description: 'Using ensemble methods to reduce variance without increasing bias',
          stepByStep: [
            'Train multiple models on different subsets of data',
            'Each model has high variance but low bias',
            'Average predictions from all models',
            'Variance reduces by factor of 1/n (n = number of models)',
            'Bias stays the same (average of low bias = low bias)'
          ],
          example: 'Random Forest: 100 decision trees, each trained on bootstrap sample. Average predictions reduces variance.',
          code: `# Ensemble to reduce variance
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# Random Forest (reduces variance)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Bagging (Bootstrap Aggregating)
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                            n_estimators=100)
bagging.fit(X_train, y_train)

# Variance reduction: Var(avg) = Var(individual) / n`,
          realWorld: 'Random forests, bagging, boosting - all reduce variance through ensemble',
          visual: 'Shows how ensemble averaging reduces prediction variance'
        },
        {
          title: 'Early Stopping to Prevent Overfitting',
          icon: <Gauge className="w-6 h-6" />,
          description: 'Stopping training when validation loss stops improving',
          stepByStep: [
            'Monitor validation loss during training',
            'If validation loss stops improving (or increases), stop training',
            'Prevents model from overfitting to training data',
            'Saves computation time',
            'Returns model with best validation performance'
          ],
          example: 'Neural network: Stop training when validation loss plateaus, even if training loss keeps decreasing.',
          code: `# Early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Wait 10 epochs without improvement
    restore_best_weights=True
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          callbacks=[early_stopping])`,
          realWorld: 'Neural networks, gradient boosting, any iterative training algorithm',
          visual: 'Shows training stopping when validation loss plateaus'
        },
        {
          title: 'Data Augmentation to Reduce Variance',
          icon: <Image className="w-6 h-6" />,
          description: 'Increasing effective training data size through augmentation',
          stepByStep: [
            'Identify data augmentation techniques (rotation, flipping, noise)',
            'Apply random augmentations during training',
            'Effectively increases training set size',
            'Reduces variance by exposing model to more variations',
            'Doesn\'t increase bias (augmentations preserve label)'
          ],
          example: 'Image classification: Rotate, flip, crop images. Model sees more variations, reduces overfitting.',
          code: `# Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Augmentation for images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Apply during training
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=100)`,
          realWorld: 'Image classification, NLP (text augmentation), any domain with augmentation techniques',
          visual: 'Shows original and augmented images'
        }
      ],
      'regularization': [
        {
          title: 'L1 Regularization for Feature Selection',
          icon: <Zap className="w-6 h-6" />,
          description: 'Using L1 (Lasso) to automatically select important features',
          stepByStep: [
            'Add L1 penalty: λ × Σ|w| to loss function',
            'L1 shrinks weights towards zero',
            'Some weights become exactly zero (feature elimination)',
            'Model automatically selects relevant features',
            'Tune λ using cross-validation'
          ],
          example: 'Predicting house prices: L1 removes irrelevant features like "number of windows"',
          code: `# L1 Regularization (Lasso)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)  # alpha = lambda
lasso.fit(X_train, y_train)

# Check which features were selected
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"Selected {len(selected_features)} features out of {X.shape[1]}")`,
          realWorld: 'Feature selection in high-dimensional data, interpretable models',
          visual: 'Shows weights shrinking to zero as lambda increases'
        },
        {
          title: 'L2 Regularization for Overfitting Prevention',
          icon: <TrendingUp className="w-6 h-6" />,
          description: 'Using L2 (Ridge) to prevent overfitting without feature elimination',
          stepByStep: [
            'Add L2 penalty: λ × Σw² to loss function',
            'L2 shrinks all weights proportionally',
            'Prevents weights from becoming too large',
            'Reduces model complexity without eliminating features',
            'Tune λ to balance bias and variance'
          ],
          example: 'Neural network: L2 regularization prevents weights from exploding',
          code: `# L2 Regularization (Ridge)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)  # alpha = lambda
ridge.fit(X_train, y_train)

# In neural networks (PyTorch)
import torch.nn as nn
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.01, 
                            weight_decay=0.01)  # L2 penalty`,
          realWorld: 'Preventing overfitting in linear models, neural networks, polynomial regression',
          visual: 'Shows weights shrinking proportionally as lambda increases'
        },
        {
          title: 'Elastic Net: Combining L1 and L2',
          icon: <Network className="w-6 h-6" />,
          description: 'Using both L1 and L2 regularization for balanced approach',
          stepByStep: [
            'Combine penalties: λ₁ × Σ|w| + λ₂ × Σw²',
            'L1 provides feature selection',
            'L2 provides smooth shrinkage',
            'Get benefits of both regularization types',
            'Tune both λ₁ and λ₂ using grid search'
          ],
          example: 'When you want feature selection but also smooth weight shrinkage',
          code: `# Elastic Net
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio balances L1 vs L2
elastic.fit(X_train, y_train)

# l1_ratio = 1.0 → Pure L1 (Lasso)
# l1_ratio = 0.0 → Pure L2 (Ridge)
# l1_ratio = 0.5 → Balanced`,
          realWorld: 'High-dimensional datasets where both feature selection and overfitting prevention needed',
          visual: 'Shows combined effect of L1 and L2 penalties'
        },
        {
          title: 'Dropout Regularization in Neural Networks',
          icon: <Network className="w-6 h-6" />,
          description: 'Using dropout to prevent overfitting in neural networks',
          stepByStep: [
            'Randomly set some neurons to zero during training',
            'Dropout rate: probability of setting neuron to zero (e.g., 0.5 = 50%)',
            'Prevents neurons from co-adapting',
            'Forces network to learn redundant representations',
            'During inference: use all neurons but scale by dropout rate'
          ],
          example: 'Deep neural network: Dropout 0.5 means each neuron has 50% chance of being dropped during training.',
          code: `# Dropout in neural networks
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)  # 30% dropout
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)  # Applied during training
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x`,
          realWorld: 'Deep neural networks, CNNs, RNNs, transformers - all benefit from dropout',
          visual: 'Shows neurons being randomly dropped during training'
        },
        {
          title: 'Weight Decay in Deep Learning',
          icon: <Settings className="w-6 h-6" />,
          description: 'Applying L2 regularization (weight decay) in neural networks',
          stepByStep: [
            'Add L2 penalty to loss function: λ × Σw²',
            'Or use weight_decay parameter in optimizer',
            'Penalizes large weights',
            'Prevents weights from growing too large',
            'Encourages simpler, more generalizable models'
          ],
          example: 'Training neural network: weight_decay=0.0001 prevents weights from exploding, improves generalization.',
          code: `# Weight decay (L2 regularization) in PyTorch
import torch.optim as optim

# Weight decay in optimizer
optimizer = optim.SGD(model.parameters(),
                     lr=0.01,
                     weight_decay=0.0001)  # L2 penalty

# Or in TensorFlow/Keras
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, weight_decay=0.0001),
    loss='categorical_crossentropy'
)`,
          realWorld: 'All deep learning: CNNs, RNNs, transformers, any neural network',
          visual: 'Shows weight distributions with and without weight decay'
        },
        {
          title: 'Batch Normalization as Regularization',
          icon: <Activity className="w-6 h-6" />,
          description: 'Using batch normalization for regularization effect',
          stepByStep: [
            'Normalize activations within each mini-batch',
            'Adds noise to activations (regularization effect)',
            'Reduces internal covariate shift',
            'Allows higher learning rates',
            'Acts as implicit regularization'
          ],
          example: 'Deep CNN: Batch normalization after each conv layer reduces overfitting and speeds up training.',
          code: `# Batch Normalization
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.bn1(nn.functional.relu(self.conv1(x)))
        x = self.bn2(nn.functional.relu(self.conv2(x)))
        return x`,
          realWorld: 'Deep CNNs, ResNets, all modern deep learning architectures',
          visual: 'Shows activations before and after batch normalization'
        }
      ]
    },
    'unsupervised-learning': {
      'clustering': [
        {
          title: 'Customer Segmentation with K-Means',
          icon: <Users className="w-6 h-6" />,
          description: 'Grouping customers based on purchasing behavior for targeted marketing',
          stepByStep: [
            'Collect customer data: purchase history, demographics, browsing behavior',
            'Preprocess and normalize features',
            'Choose number of clusters (k) using elbow method or domain knowledge',
            'Apply K-means clustering',
            'Analyze cluster characteristics',
            'Create targeted marketing campaigns for each segment'
          ],
          example: 'E-commerce: Segment customers into "bargain hunters", "premium buyers", "occasional shoppers"',
          code: `# Customer Segmentation with K-Means
from sklearn.cluster import KMeans
import pandas as pd

# Load customer data
data = pd.read_csv('customers.csv')
features = ['annual_spending', 'frequency', 'avg_order_value']

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Analyze clusters
data['cluster'] = clusters
print(data.groupby('cluster')[features].mean())`,
          realWorld: 'Marketing, e-commerce, customer relationship management, retail analytics',
          visual: 'Shows customer data points grouped into distinct clusters'
        },
        {
          title: 'Image Segmentation with Hierarchical Clustering',
          icon: <Image className="w-6 h-6" />,
          description: 'Grouping pixels in images to identify objects or regions',
          stepByStep: [
            'Extract pixel features (color, position)',
            'Build distance matrix between pixels',
            'Apply hierarchical clustering',
            'Cut dendrogram at desired number of segments',
            'Visualize segmented image'
          ],
          example: 'Medical imaging: Segment brain MRI to identify different tissue types',
          code: `# Image Segmentation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from PIL import Image

# Load and preprocess image
img = Image.open('image.jpg')
pixels = np.array(img).reshape(-1, 3)  # Flatten to (pixels, RGB)

# Apply hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=5)
labels = clustering.fit_predict(pixels)

# Reshape labels back to image dimensions
segmented = labels.reshape(img.size[1], img.size[0])`,
          realWorld: 'Computer vision, medical imaging, satellite image analysis, object detection',
          visual: 'Shows original image and segmented regions with different colors'
        },
        {
          title: 'Anomaly Detection in Network Traffic with DBSCAN',
          icon: <Network className="w-6 h-6" />,
          description: 'Identifying unusual network patterns that might indicate security threats',
          stepByStep: [
            'Collect network traffic features (packet size, frequency, destination)',
            'Normalize features',
            'Apply DBSCAN clustering',
            'Points marked as noise are potential anomalies',
            'Investigate anomalies for security threats'
          ],
          example: 'Cybersecurity: Detect DDoS attacks, port scanning, or data exfiltration',
          code: `# Network Anomaly Detection with DBSCAN
from sklearn.cluster import DBSCAN
import numpy as np

# Network traffic features
features = ['packet_size', 'packet_rate', 'unique_destinations']

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Anomalies are points with cluster label -1
anomalies = X[clusters == -1]
print(f"Found {len(anomalies)} potential anomalies")`,
          realWorld: 'Network security, intrusion detection, fraud detection, system monitoring',
          visual: 'Shows normal traffic clusters and isolated anomaly points'
        }
      ],
      'dimensionality-reduction': [
        {
          title: 'Face Recognition with PCA',
          icon: <Users className="w-6 h-6" />,
          description: 'Reducing face image dimensions while preserving identity information',
          stepByStep: [
            'Collect face images dataset',
            'Flatten images to vectors (e.g., 64x64 → 4096 dimensions)',
            'Center the data (subtract mean face)',
            'Compute covariance matrix',
            'Find principal components (eigenfaces)',
            'Project faces onto lower-dimensional space',
            'Use reduced features for recognition'
          ],
          example: 'Security systems: Reduce 4096-dim face images to 50 principal components for efficient matching',
          code: `# Face Recognition with PCA
from sklearn.decomposition import PCA
import numpy as np

# Load face images (each row is a flattened image)
faces = load_faces()  # Shape: (n_samples, 4096)

# Apply PCA
pca = PCA(n_components=50)
faces_reduced = pca.fit_transform(faces)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Recognition: project new face and compare in reduced space
new_face_reduced = pca.transform(new_face)`,
          realWorld: 'Biometrics, security systems, image compression, computer vision',
          visual: 'Shows original faces and eigenfaces (principal components)'
        },
        {
          title: 'Visualizing High-Dimensional Data with t-SNE',
          icon: <BarChart className="w-6 h-6" />,
          description: 'Reducing dimensions to 2D/3D for visualization and exploration',
          stepByStep: [
            'Prepare high-dimensional data (e.g., word embeddings, gene expressions)',
            'Choose perplexity parameter (typically 5-50)',
            'Apply t-SNE to reduce to 2D or 3D',
            'Visualize reduced data',
            'Interpret clusters and patterns'
          ],
          example: 'Genomics: Visualize 20,000-dimensional gene expression data in 2D to identify cell types',
          code: `# t-SNE Visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# High-dimensional data (e.g., word embeddings)
X = load_embeddings()  # Shape: (n_samples, 300)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

# Visualize
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
plt.title('t-SNE Visualization')
plt.show()`,
          realWorld: 'Data exploration, genomics, NLP, recommendation systems, bioinformatics',
          visual: 'Shows high-dimensional data projected to 2D with preserved local structure'
        },
        {
          title: 'Feature Extraction with Autoencoders',
          icon: <Brain className="w-6 h-6" />,
          description: 'Learning compressed representations for downstream tasks',
          stepByStep: [
            'Design autoencoder architecture (encoder-decoder)',
            'Train to reconstruct input from compressed representation',
            'Extract encoder output as features',
            'Use compressed features for classification or clustering',
            'Fine-tune if needed'
          ],
          example: 'Image classification: Compress 784-dim MNIST images to 32-dim features, then classify',
          code: `# Autoencoder for Feature Extraction
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train autoencoder
model = Autoencoder()
# ... training code ...

# Extract features
features = model.encoder(images)  # Shape: (batch, 32)`,
          realWorld: 'Deep learning, image processing, NLP, anomaly detection, data compression',
          visual: 'Shows original and reconstructed images, and learned feature space'
        }
      ],
      'anomaly-detection': [
        {
          title: 'Credit Card Fraud Detection',
          icon: <CreditCard className="w-6 h-6" />,
          description: 'Identifying fraudulent transactions in real-time',
          stepByStep: [
            'Collect transaction features: amount, time, location, merchant type',
            'Train on historical normal transactions',
            'Apply Isolation Forest or One-Class SVM',
            'Flag transactions with high anomaly scores',
            'Review flagged transactions',
            'Update model with feedback'
          ],
          example: 'Banking: Detect unusual spending patterns like $5000 purchase at 3 AM in foreign country',
          code: `# Fraud Detection with Isolation Forest
from sklearn.ensemble import IsolationForest
import pandas as pd

# Transaction features
features = ['amount', 'hour', 'merchant_category', 'distance_from_home']

# Train on normal transactions
normal_transactions = load_normal_transactions()
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(normal_transactions[features])

# Detect anomalies in new transactions
new_transactions = load_new_transactions()
predictions = model.predict(new_transactions[features])
fraudulent = new_transactions[predictions == -1]`,
          realWorld: 'Banking, e-commerce, insurance, financial services, payment processing',
          visual: 'Shows normal transaction clusters and isolated fraudulent transactions'
        },
        {
          title: 'Manufacturing Quality Control',
          icon: <Settings className="w-6 h-6" />,
          description: 'Detecting defective products on production line',
          stepByStep: [
            'Collect sensor data: temperature, pressure, vibration, dimensions',
            'Train on normal production data',
            'Apply anomaly detection algorithm',
            'Flag products with unusual sensor readings',
            'Remove defective products from line',
            'Analyze patterns in defects'
          ],
          example: 'Automotive: Detect engine parts with abnormal vibration patterns',
          code: `# Quality Control with One-Class SVM
from sklearn.svm import OneClassSVM
import numpy as np

# Sensor data from production line
sensor_data = load_sensor_data()  # Features: temp, pressure, vibration

# Train on normal products
normal_products = sensor_data[sensor_data['label'] == 'normal']
model = OneClassSVM(nu=0.05, kernel='rbf')
model.fit(normal_products[['temp', 'pressure', 'vibration']])

# Detect defects
predictions = model.predict(sensor_data[['temp', 'pressure', 'vibration']])
defects = sensor_data[predictions == -1]`,
          realWorld: 'Manufacturing, quality assurance, predictive maintenance, industrial IoT',
          visual: 'Shows normal product distribution and isolated defect points'
        },
        {
          title: 'Network Intrusion Detection',
          icon: <Shield className="w-6 h-6" />,
          description: 'Identifying malicious network activity',
          stepByStep: [
            'Collect network flow features: bytes, packets, duration, protocol',
            'Train on normal network traffic',
            'Apply anomaly detection',
            'Flag suspicious connections',
            'Investigate flagged connections',
            'Update detection rules'
          ],
          example: 'Cybersecurity: Detect port scanning, DDoS attacks, or unauthorized access attempts',
          code: `# Network Intrusion Detection
from sklearn.ensemble import IsolationForest

# Network flow features
features = ['src_bytes', 'dst_bytes', 'duration', 'protocol_type']

# Train on normal traffic
normal_traffic = load_normal_traffic()
model = IsolationForest(contamination=0.02)
model.fit(normal_traffic[features])

# Detect intrusions
new_traffic = load_new_traffic()
anomaly_scores = model.decision_function(new_traffic[features])
intrusions = new_traffic[anomaly_scores < threshold]`,
          realWorld: 'Cybersecurity, network monitoring, IT security, system administration',
          visual: 'Shows normal network traffic clusters and isolated intrusion attempts'
        }
      ],
      'distance-metrics': [
        {
          title: 'Document Similarity Search',
          icon: <FileText className="w-6 h-6" />,
          description: 'Finding similar documents using cosine similarity',
          stepByStep: [
            'Convert documents to TF-IDF vectors',
            'Calculate cosine similarity between query and all documents',
            'Rank documents by similarity',
            'Return top-k most similar documents',
            'Display results'
          ],
          example: 'Search engine: Find news articles similar to a query article',
          code: `# Document Similarity with Cosine Distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert documents to vectors
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Query document
query_vector = vectorizer.transform([query_doc])

# Calculate cosine similarities
similarities = cosine_similarity(query_vector, doc_vectors)[0]

# Get top-k similar documents
top_k_indices = np.argsort(similarities)[-k:][::-1]
similar_docs = [documents[i] for i in top_k_indices]`,
          realWorld: 'Search engines, recommendation systems, plagiarism detection, content matching',
          visual: 'Shows document vectors and similarity scores'
        },
        {
          title: 'Image Retrieval with Euclidean Distance',
          icon: <Image className="w-6 h-6" />,
          description: 'Finding similar images based on feature vectors',
          stepByStep: [
            'Extract image features (e.g., using CNN)',
            'Store feature vectors in database',
            'Extract features from query image',
            'Calculate Euclidean distance to all images',
            'Return images with smallest distances'
          ],
          example: 'E-commerce: Find products with similar visual appearance',
          code: `# Image Retrieval with Euclidean Distance
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Extract features (e.g., using pre-trained CNN)
def extract_features(image):
    # ... CNN feature extraction ...
    return features  # Shape: (2048,)

# Database of image features
image_features = [extract_features(img) for img in database_images]

# Query image
query_features = extract_features(query_image)

# Calculate distances
distances = euclidean_distances([query_features], image_features)[0]

# Get similar images
similar_indices = np.argsort(distances)[:k]
similar_images = [database_images[i] for i in similar_indices]`,
          realWorld: 'Image search, e-commerce, content-based recommendation, visual similarity',
          visual: 'Shows query image and retrieved similar images with distances'
        },
        {
          title: 'Clustering with Manhattan Distance',
          icon: <Layers className="w-6 h-6" />,
          description: 'Using Manhattan distance for robust clustering',
          stepByStep: [
            'Choose distance metric (Manhattan for robustness)',
            'Calculate distance matrix between all points',
            'Apply clustering algorithm (K-means with Manhattan)',
            'Evaluate cluster quality',
            'Interpret results'
          ],
          example: 'Customer segmentation: Group customers using Manhattan distance to handle outliers',
          code: `# K-Means with Manhattan Distance
from sklearn.cluster import KMeans
from scipy.spatial.distance import cityblock

# Custom distance metric for K-means
def manhattan_distance(x, y):
    return cityblock(x, y)

# Apply clustering
kmeans = KMeans(n_clusters=3, metric=manhattan_distance)
clusters = kmeans.fit_predict(data)

# Or use Manhattan distance in hierarchical clustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

distances = pdist(data, metric='cityblock')
linkage_matrix = linkage(distances, method='ward')
clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')`,
          realWorld: 'Clustering algorithms, robust data analysis, feature selection, outlier handling',
          visual: 'Shows clustering results with Manhattan distance paths'
        }
      ]
    },
    'neural-networks': {
      'architecture': [
        {
          title: 'Image Classification with Convolutional Neural Networks',
          icon: <Image className="w-6 h-6" />,
          description: 'Using CNN architecture to classify images into categories',
          stepByStep: [
            'Design CNN architecture: Convolutional layers → Pooling → Fully connected',
            'Input layer receives image pixels (e.g., 224×224×3 for RGB)',
            'Convolutional layers extract features (edges, shapes, patterns)',
            'Pooling layers reduce spatial dimensions',
            'Fully connected layers make final classification',
            'Output layer produces class probabilities'
          ],
          example: 'Classifying cats vs dogs: CNN learns to recognize fur patterns, ear shapes, etc.',
          code: `# CNN Architecture for Image Classification
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x`,
          realWorld: 'Image recognition, medical imaging, autonomous vehicles, facial recognition',
          visual: 'Shows CNN architecture with convolutional and fully connected layers'
        },
        {
          title: 'Natural Language Processing with Recurrent Neural Networks',
          icon: <MessageSquare className="w-6 h-6" />,
          description: 'Using RNN architecture to process sequential text data',
          stepByStep: [
            'Design RNN architecture: Input → Hidden → Output',
            'Input layer receives word embeddings',
            'Hidden layer maintains memory of previous words',
            'Recurrent connections pass information forward',
            'Output layer predicts next word or classifies text',
            'LSTM/GRU variants handle long sequences better'
          ],
          example: 'Sentiment analysis: RNN processes words sequentially to understand context',
          code: `# RNN for Text Classification
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])`,
          realWorld: 'Text classification, machine translation, speech recognition, chatbots',
          visual: 'Shows RNN architecture with recurrent connections'
        }
      ],
      'forward-pass': [
        {
          title: 'Real-Time Prediction with Neural Networks',
          icon: <Zap className="w-6 h-6" />,
          description: 'Using forward pass for fast inference in production systems',
          stepByStep: [
            'Load trained model weights',
            'Preprocess input data (normalize, encode)',
            'Forward pass: Input → Hidden layers → Output',
            'Apply activation functions at each layer',
            'Get prediction from output layer',
            'Post-process results (decode, format)'
          ],
          example: 'Recommendation system: Forward pass through neural network to predict user preferences in milliseconds',
          code: `# Forward Pass for Inference
import torch

def predict(model, input_data):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # No gradient computation
        # Forward pass
        output = model(input_data)
        # Get predictions
        predictions = torch.softmax(output, dim=1)
        return predictions

# Usage
model = load_trained_model()
user_features = preprocess_user_data(user_id)
recommendations = predict(model, user_features)`,
          realWorld: 'Real-time recommendations, fraud detection, autonomous driving, medical diagnosis',
          visual: 'Shows data flowing through network layers during forward pass'
        }
      ],
      'backpropagation': [
        {
          title: 'Training Deep Neural Networks',
          icon: <Brain className="w-6 h-6" />,
          description: 'Using backpropagation to train complex models',
          stepByStep: [
            'Forward pass: Compute predictions',
            'Calculate loss: Compare predictions to targets',
            'Backpropagation: Compute gradients using chain rule',
            'Gradient flows backward through all layers',
            'Update weights: w = w - α × gradient',
            'Repeat for multiple epochs'
          ],
          example: 'Training image classifier: Backpropagation adjusts millions of weights to minimize classification error',
          code: `# Training Loop with Backpropagation
import torch
import torch.nn as nn
import torch.optim as optim

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass (backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()`,
          realWorld: 'Training all deep learning models: CNNs, RNNs, transformers, GANs',
          visual: 'Shows gradients flowing backward through network layers'
        }
      ],
      'activation-functions': [
        {
          title: 'Choosing Activation Functions for Different Tasks',
          icon: <Target className="w-6 h-6" />,
          description: 'Selecting appropriate activation functions based on problem type',
          stepByStep: [
            'For hidden layers: Use ReLU (most common) or variants',
            'For binary classification output: Use Sigmoid',
            'For multi-class classification: Use Softmax',
            'For regression output: Use Linear (no activation)',
            'Consider gradient flow: ReLU avoids vanishing gradients',
            'Experiment with different activations'
          ],
          example: 'Image classification: ReLU in hidden layers, Softmax in output layer',
          code: `# Activation Functions in Neural Network
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes
    
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))  # ReLU for hidden
        x = nn.ReLU()(self.fc2(x))  # ReLU for hidden
        x = nn.Softmax(dim=1)(self.fc3(x))  # Softmax for output
        return x`,
          realWorld: 'All neural network architectures require appropriate activation functions',
          visual: 'Shows different activation function curves and their derivatives'
        }
      ],
      'transformers': [
        {
          title: 'Building Large Language Models (LLMs)',
          icon: <MessageSquare className="w-6 h-6" />,
          description: 'Using transformer architecture to build GPT-like language models',
          stepByStep: [
            'Tokenize text into tokens',
            'Create embeddings for tokens',
            'Add positional encoding',
            'Apply multi-head self-attention',
            'Use feed-forward networks',
            'Stack multiple transformer blocks',
            'Pre-train on large text corpus',
            'Fine-tune for specific tasks'
          ],
          example: 'GPT-3: 175 billion parameters, trained on internet text, generates human-like text',
          code: `# Transformer Architecture (Simplified)
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x`,
          realWorld: 'GPT, BERT, T5, ChatGPT, language translation, text generation, code completion',
          visual: 'Shows transformer architecture with attention mechanisms'
        },
        {
          title: 'Machine Translation with Encoder-Decoder Transformers',
          icon: <Network className="w-6 h-6" />,
          description: 'Using transformer architecture for sequence-to-sequence tasks',
          stepByStep: [
            'Encoder processes source language tokens',
            'Self-attention captures relationships in source',
            'Decoder generates target language tokens',
            'Cross-attention connects encoder and decoder',
            'Generate translations token by token',
            'Use beam search for better translations'
          ],
          example: 'Google Translate: Encoder reads English, decoder generates Spanish',
          code: `# Encoder-Decoder Transformer
class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model)
        self.decoder = TransformerDecoder(tgt_vocab, d_model)
    
    def forward(self, src, tgt):
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out)
        return decoder_out`,
          realWorld: 'Machine translation, summarization, question answering, text-to-speech',
          visual: 'Shows encoder-decoder architecture with attention connections'
        }
      ],
      'training': [
        {
          title: 'Training Large Language Models',
          icon: <Brain className="w-6 h-6" />,
          description: 'Training process for GPT-style models on massive datasets',
          stepByStep: [
            'Collect large text corpus (billions of tokens)',
            'Tokenize and preprocess text',
            'Initialize transformer architecture',
            'Pre-train using self-supervised learning (predict next token)',
            'Use distributed training across multiple GPUs',
            'Monitor loss and adjust hyperparameters',
            'Fine-tune on specific tasks',
            'Deploy and serve model'
          ],
          example: 'GPT-3 training: 175B parameters, trained on 300B tokens, cost millions of dollars',
          code: `# Training Large Language Model (Simplified)
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        
        # Forward pass
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`,
          realWorld: 'Training GPT, BERT, T5, and other foundation models',
          visual: 'Shows loss decreasing over training epochs'
        },
        {
          title: 'Transfer Learning and Fine-Tuning',
          icon: <Layers className="w-6 h-6" />,
          description: 'Using pre-trained models and adapting them for specific tasks',
          stepByStep: [
            'Load pre-trained model weights',
            'Freeze early layers (keep learned features)',
            'Replace or fine-tune final layers',
            'Train on task-specific dataset',
            'Use smaller learning rate',
            'Monitor validation performance',
            'Deploy fine-tuned model'
          ],
          example: 'Fine-tuning BERT for sentiment analysis: Use pre-trained BERT, add classification head, train on reviews',
          code: `# Fine-tuning Pre-trained Model
from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Freeze early layers (optional)
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# Fine-tune on your data
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()`,
          realWorld: 'Adapting GPT for code completion, fine-tuning BERT for domain-specific tasks',
          visual: 'Shows pre-trained model being adapted for new task'
        }
      ]
    }
  };

  const currentUseCases = useCasesByDomain[domain]?.[operationType] || [];

  if (currentUseCases.length === 0) return null;

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg p-6 border-2 border-indigo-200">
      <h3 className="text-2xl font-bold text-indigo-900 mb-6 flex items-center gap-2">
        <Brain className="w-7 h-7" />
        Real-World ML Applications
      </h3>
      
      <div className="space-y-4">
        {currentUseCases.map((useCase, idx) => (
          <div
            key={idx}
            className="bg-white rounded-lg border-2 border-gray-200 hover:border-indigo-300 transition-all overflow-hidden"
          >
            <button
              onClick={() => setExpandedCase(expandedCase === idx ? null : idx)}
              className="w-full p-5 flex items-start gap-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex-shrink-0 w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center text-indigo-600">
                {useCase.icon}
              </div>
              <div className="flex-1 text-left">
                <h4 className="text-lg font-bold text-gray-900 mb-1">
                  {useCase.title}
                </h4>
                <p className="text-sm text-gray-600">
                  {useCase.description}
                </p>
              </div>
              <div className="flex-shrink-0">
                {expandedCase === idx ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </div>
            </button>

            {expandedCase === idx && (
              <div className="px-5 pb-5 space-y-4 border-t border-gray-200 pt-4">
                {/* Step-by-Step Guide */}
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h5 className="font-bold text-blue-900 mb-3 flex items-center gap-2">
                    <Layers className="w-5 h-5" />
                    Step-by-Step Process
                  </h5>
                  <ol className="space-y-2">
                    {useCase.stepByStep.map((step, stepIdx) => (
                      <li key={stepIdx} className="flex gap-3">
                        <span className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          {stepIdx + 1}
                        </span>
                        <span className="text-sm text-blue-900 flex-1">{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>

                {/* Example */}
                <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-200">
                  <h5 className="font-semibold text-indigo-900 mb-2 text-sm">Example:</h5>
                  <p className="text-sm text-indigo-800 font-mono bg-white px-3 py-2 rounded">
                    {useCase.example}
                  </p>
                </div>

                {/* Code */}
                <div className="bg-gray-900 rounded-lg p-4 border-2 border-gray-700">
                  <h5 className="text-xs text-gray-400 mb-2 font-semibold">Code Example:</h5>
                  <pre className="text-xs text-green-400 overflow-x-auto">
                    {useCase.code}
                  </pre>
                </div>

                {/* Real-World Usage */}
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <h5 className="font-semibold text-green-900 mb-2 text-sm flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Real-World Usage:
                  </h5>
                  <p className="text-sm text-green-800">
                    {useCase.realWorld}
                  </p>
                </div>

                {/* Visual Aid Description */}
                {useCase.visual && (
                  <div className="bg-yellow-50 rounded-lg p-3 border border-yellow-200">
                    <p className="text-xs text-yellow-900">
                      <strong>Visual Aid:</strong> {useCase.visual}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

