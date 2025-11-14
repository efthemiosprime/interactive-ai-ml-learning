import React from 'react';
import { Link } from 'react-router-dom';
import { Calculator, Brain, BarChart3, Target, Layers, Network, Code, Rocket, ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      <div className="max-w-6xl mx-auto px-8 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 mb-4">
            AI & Machine Learning Math Tutorial
          </h1>
          <p className="text-xl text-gray-700 max-w-2xl mx-auto">
            Interactive visualizations and explanations for the mathematical foundations of AI and ML: 
            Linear Algebra, Calculus, Probability & Statistics, Supervised Learning, Unsupervised Learning, 
            Neural Networks, Programming Tutorials, and Real-World AI/ML Applications
          </p>
        </div>

        {/* Table of Contents */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {/* Linear Algebra Card */}
          <Link
            to="/linear-algebra"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-indigo-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Calculator className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-indigo-900 mb-3 group-hover:text-indigo-700 transition-colors">
                  Linear Algebra
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Master eigenvalues and eigenvectors, matrix operations, and understand how data and weights 
                  are represented in ML models. Essential for understanding neural networks and data transformations.
                </p>
                <div className="flex items-center gap-2 text-indigo-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                  Eigenvalues & Eigenvectors
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                  Matrix Operations for ML
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                  Data & Weight Representation
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></span>
                  Interactive Visualizations
                </li>
              </ul>
            </div>
          </Link>

          {/* Calculus Card */}
          <Link
            to="/calculus"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-purple-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-purple-900 mb-3 group-hover:text-purple-700 transition-colors">
                  Calculus
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Learn derivatives, partial derivatives, gradients, and the chain rule. Understand how these 
                  concepts power backpropagation in neural networks and optimization algorithms.
                </p>
                <div className="flex items-center gap-2 text-purple-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Derivatives & Partial Derivatives
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Gradients & Gradient Descent
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Chain Rule & Backpropagation
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Step-by-Step Explanations
                </li>
              </ul>
            </div>
          </Link>

          {/* Probability & Statistics Card */}
          <Link
            to="/probability-statistics"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-blue-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <BarChart3 className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-blue-900 mb-3 group-hover:text-blue-700 transition-colors">
                  Probability & Statistics
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Master mean, variance, covariance, standard deviation, conditional probability, Bayes' theorem, 
                  and probability distributions. Essential for understanding ML algorithms and data analysis.
                </p>
                <div className="flex items-center gap-2 text-blue-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                  Descriptive Statistics
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                  Conditional Probability & Bayes' Theorem
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                  Probability Distributions
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                  Interactive Visualizations
                </li>
              </ul>
            </div>
          </Link>

          {/* Supervised Learning Card */}
          <Link
            to="/supervised-learning"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-green-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Target className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-green-900 mb-3 group-hover:text-green-700 transition-colors">
                  Supervised Learning
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Learn loss functions, model evaluation metrics, bias-variance tradeoff, and regularization techniques. 
                  Essential concepts for building and evaluating ML models.
                </p>
                <div className="flex items-center gap-2 text-green-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                  Loss Functions
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                  Model Evaluation Metrics
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                  Bias-Variance Tradeoff
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                  Regularization (L1/L2)
                </li>
              </ul>
            </div>
          </Link>

          {/* Unsupervised Learning Card */}
          <Link
            to="/unsupervised-learning"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-teal-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Layers className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-teal-900 mb-3 group-hover:text-teal-700 transition-colors">
                  Unsupervised Learning
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Discover patterns in unlabeled data through clustering, dimensionality reduction, and anomaly detection. 
                  Learn how distance metrics and linear algebra power these algorithms.
                </p>
                <div className="flex items-center gap-2 text-teal-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-teal-500 rounded-full"></span>
                  Clustering (K-means, Hierarchical, DBSCAN)
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-teal-500 rounded-full"></span>
                  Dimensionality Reduction (PCA, t-SNE)
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-teal-500 rounded-full"></span>
                  Anomaly Detection
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-teal-500 rounded-full"></span>
                  Distance Metrics
                </li>
              </ul>
            </div>
          </Link>

          {/* Neural Networks & Deep Learning Card */}
          <Link
            to="/neural-networks"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-violet-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-violet-500 to-fuchsia-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Network className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-violet-900 mb-3 group-hover:text-violet-700 transition-colors">
                  Neural Networks & Deep Learning
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Master neural network architecture, forward pass, backpropagation, activation functions, 
                  and transformers. Essential for understanding modern AI and building LLMs.
                </p>
                <div className="flex items-center gap-2 text-violet-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-violet-500 rounded-full"></span>
                  Neural Network Architecture
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-violet-500 rounded-full"></span>
                  Forward Pass & Backpropagation
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-violet-500 rounded-full"></span>
                  Activation Functions
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-violet-500 rounded-full"></span>
                  Transformers & Attention
                </li>
              </ul>
            </div>
          </Link>

          {/* Programming Tutorial Card */}
          <Link
            to="/programming-tutorial"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-orange-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Code className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-orange-900 mb-3 group-hover:text-orange-700 transition-colors">
                  Programming Tutorial
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Hands-on programming exercises with PyTorch and TensorFlow. Build models from scratch, 
                  use pre-trained models, and implement complete ML pipelines.
                </p>
                <div className="flex items-center gap-2 text-orange-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-orange-500 rounded-full"></span>
                  PyTorch & TensorFlow Basics
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-orange-500 rounded-full"></span>
                  Building Models from Scratch
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-orange-500 rounded-full"></span>
                  Pre-trained Models & Transfer Learning
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-orange-500 rounded-full"></span>
                  Complete Training Pipelines
                </li>
              </ul>
            </div>
          </Link>

          {/* AI/ML Applications Card */}
          <Link
            to="/ai-ml-applications"
            className="group bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border-2 border-transparent hover:border-purple-300"
          >
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <Rocket className="w-8 h-8 text-white" />
              </div>
              <div className="flex-1">
                <h2 className="text-3xl font-bold text-purple-900 mb-3 group-hover:text-purple-700 transition-colors">
                  Real-World AI/ML Applications
                </h2>
                <p className="text-gray-600 mb-4 leading-relaxed">
                  Complete end-to-end tutorials building real AI/ML applications. Put all concepts together 
                  into practical projects: image classification, NLP, recommendation systems, and more.
                </p>
                <div className="flex items-center gap-2 text-purple-600 font-semibold group-hover:gap-4 transition-all">
                  <span>Explore</span>
                  <ArrowRight className="w-5 h-5" />
                </div>
              </div>
            </div>
            
            {/* Features List */}
            <div className="mt-6 pt-6 border-t border-gray-200">
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Image Classification (CNNs)
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Sentiment Analysis (NLP)
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Recommendation Systems
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                  Complete End-to-End Projects
                </li>
              </ul>
            </div>
          </Link>
        </div>

        {/* Additional Info Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8 border-t-4 border-indigo-500">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">About This Platform</h3>
          <p className="text-gray-700 leading-relaxed mb-4">
            This interactive learning platform combines mathematical theory with visual programming to help you 
            understand the mathematical foundations of AI and Machine Learning through hands-on exploration. Each section includes:
          </p>
          <div className="grid md:grid-cols-3 gap-6 mt-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-2xl">📊</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Interactive Visualizations</h4>
              <p className="text-sm text-gray-600">See math in action with real-time graphics</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-2xl">📚</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Step-by-Step Explanations</h4>
              <p className="text-sm text-gray-600">Detailed breakdowns of every calculation</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-pink-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-2xl">🤖</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">ML Applications</h4>
              <p className="text-sm text-gray-600">Real-world connections to AI/ML algorithms</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

