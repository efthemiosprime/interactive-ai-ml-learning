import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import Home from './pages/Home';
import LinearAlgebra from './pages/LinearAlgebra';
import Calculus from './pages/Calculus';
import ProbabilityStatistics from './pages/ProbabilityStatistics';
import SupervisedLearning from './pages/SupervisedLearning';
import UnsupervisedLearning from './pages/UnsupervisedLearning';
import NeuralNetworks from './pages/NeuralNetworks';
import ProgrammingTutorial from './pages/ProgrammingTutorial';
import AIMLApplications from './pages/AIMLApplications';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/linear-algebra" element={<LinearAlgebra />} />
        <Route path="/calculus" element={<Calculus />} />
        <Route path="/probability-statistics" element={<ProbabilityStatistics />} />
            <Route path="/supervised-learning" element={<SupervisedLearning />} />
            <Route path="/unsupervised-learning" element={<UnsupervisedLearning />} />
            <Route path="/neural-networks" element={<NeuralNetworks />} />
            <Route path="/programming-tutorial" element={<ProgrammingTutorial />} />
            <Route path="/ai-ml-applications" element={<AIMLApplications />} />
      </Routes>
    </div>
  );
}

export default App;

