# AI & Machine Learning Math Tutorial

An interactive educational platform for learning the mathematical foundations of AI and Machine Learning. Built with React, Vite, and TailwindCSS.

## Features

### 📊 Linear Algebra
- **Eigenvalues & Eigenvectors**: Understand how data transformations work
- **Data Representation**: Learn how data is represented as matrices in ML
- **Weight Representation**: Understand how neural network weights are stored
- **Matrix Operations**: Master matrix operations used in forward propagation

### 🧮 Calculus
- **Derivatives**: Learn how derivatives power optimization
- **Partial Derivatives**: Understand multi-variable functions
- **Gradients**: Master gradient descent optimization
- **Chain Rule**: Foundation of backpropagation
- **Backpropagation**: How neural networks learn

### 📈 Probability & Statistics
- **Descriptive Statistics**: Mean, variance, standard deviation
- **Covariance & Correlation**: Feature relationships
- **Conditional Probability**: Modeling dependencies
- **Bayes' Theorem**: Foundation of Bayesian methods
- **Probability Distributions**: Normal, Bernoulli, and more

## Getting Started

### Prerequisites
- Node.js 16+ and npm

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser to `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
ai-ml-tutorial/
├── src/
│   ├── components/
│   │   ├── linearAlgebra/      # Linear algebra components
│   │   ├── calculus/           # Calculus components
│   │   └── probabilityStatistics/  # Probability & stats components
│   ├── pages/                  # Route pages
│   ├── utils/                  # Mathematical utilities
│   ├── App.jsx                 # Main app component
│   └── main.jsx                # Entry point
├── package.json
└── vite.config.js
```

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Routing
- **TailwindCSS** - Styling
- **Lucide React** - Icons

## Educational Approach

Each topic includes:
- **Interactive Controls**: Adjust parameters and see results
- **Visualizations**: See mathematical concepts in action
- **Educational Panels**: Detailed explanations and ML applications
- **Step-by-Step**: Break down complex calculations

## ML Applications Covered

- Principal Component Analysis (PCA)
- Neural Network Training
- Gradient Descent Optimization
- Backpropagation
- Naive Bayes Classification
- Feature Engineering
- Data Preprocessing

## License

MIT

