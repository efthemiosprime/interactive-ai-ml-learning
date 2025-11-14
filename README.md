# Interactive AI & ML Learning

An interactive learning platform that teaches AI and Machine Learning concepts through visualizations, tutorials, and hands-on applications.

## Features

- **Mathematical Foundations**: Linear Algebra, Calculus, Probability & Statistics
- **Machine Learning Concepts**: Supervised Learning, Unsupervised Learning, Neural Networks
- **Complete Tutorials**: 
  - NBA Basketball Chatbot
  - Maze Solver (Pathfinding)
  - AI Trading Tools
  - Recommendation System
- **Interactive Visualizations**: Gradient Descent, Neural Networks, PCA, Convolution, K-Means Clustering
- **Real-World Applications**: Image Classification, Sentiment Analysis, Object Detection, and more

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Deployment

This project is automatically deployed to GitHub Pages using GitHub Actions.

### Manual Deployment

If you need to deploy manually:

1. Build the project:
```bash
npm run build
```

2. Deploy to GitHub Pages:
```bash
npm run deploy
```

Note: You'll need to install `gh-pages` as a dev dependency:
```bash
npm install --save-dev gh-pages
```

## GitHub Pages Setup

1. Go to your repository settings on GitHub
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "GitHub Actions"
4. The site will be automatically deployed when you push to the `main` branch

The site will be available at: `https://efthemiosprime.github.io/interactive-ai-ml-learning/`

## Technologies Used

- React 18
- Vite
- React Router
- Tailwind CSS
- React Syntax Highlighter
- Lucide React Icons

## License

MIT
