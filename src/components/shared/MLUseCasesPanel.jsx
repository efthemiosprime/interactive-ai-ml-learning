import React, { useState } from 'react';
import { Brain, Network, Image, MessageSquare, TrendingUp, Layers, Zap, Target, BarChart, Users, CreditCard, Settings, Shield, FileText } from 'lucide-react';
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

