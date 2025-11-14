import React from 'react';
import ImageClassification from './ImageClassification';
import SentimentAnalysis from './SentimentAnalysis';
import RecommendationSystem from './RecommendationSystem';
import TimeSeriesForecasting from './TimeSeriesForecasting';
import ObjectDetection from './ObjectDetection';
import TextGeneration from './TextGeneration';
import AnomalyDetection from './AnomalyDetection';
import ImageGeneration from './ImageGeneration';
import SpeechRecognition from './SpeechRecognition';
import MachineTranslation from './MachineTranslation';
import ImageSegmentation from './ImageSegmentation';
import ReinforcementLearning from './ReinforcementLearning';
import TransferLearning from './TransferLearning';
import PretrainedModels from './PretrainedModels';
import NBAChatbotTutorial from './NBAChatbotTutorial';
import MazeSolver from './MazeSolver';
import GradientDescentVisualization from './GradientDescentVisualization';
import NeuralNetworkPlayground from './NeuralNetworkPlayground';
import LinearRegressionVisualization from './LinearRegressionVisualization';
import PCAVisualization from './PCAVisualization';
import ConvolutionVisualization from './ConvolutionVisualization';
import KMeansClusteringVisualization from './KMeansClusteringVisualization';
import TradingToolsTutorial from './TradingToolsTutorial';

export default function AIMLApplicationsVisualization({ selectedApplication }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Complete Application Code</h2>

      {selectedApplication === 'image-classification' && (
        <ImageClassification />
      )}

      {selectedApplication === 'sentiment-analysis' && (
        <SentimentAnalysis />
      )}

      {selectedApplication === 'recommendation-system' && (
        <RecommendationSystem />
      )}

      {selectedApplication === 'time-series-forecasting' && (
        <TimeSeriesForecasting />
      )}

      {selectedApplication === 'object-detection' && (
        <ObjectDetection />
      )}

      {selectedApplication === 'text-generation' && (
        <TextGeneration />
      )}

      {selectedApplication === 'anomaly-detection' && (
        <AnomalyDetection />
      )}

      {selectedApplication === 'image-generation' && (
        <ImageGeneration />
      )}

      {selectedApplication === 'speech-recognition' && (
        <SpeechRecognition />
      )}

      {selectedApplication === 'machine-translation' && (
        <MachineTranslation />
      )}

      {selectedApplication === 'image-segmentation' && (
        <ImageSegmentation />
      )}

      {selectedApplication === 'reinforcement-learning' && (
        <ReinforcementLearning />
      )}

      {selectedApplication === 'transfer-learning' && (
        <TransferLearning />
      )}

      {selectedApplication === 'pretrained-models' && (
        <PretrainedModels />
      )}

      {selectedApplication === 'nba-chatbot' && (
        <NBAChatbotTutorial />
      )}

      {selectedApplication === 'maze-solver' && (
        <MazeSolver />
      )}

      {selectedApplication === 'gradient-descent' && (
        <GradientDescentVisualization />
      )}

      {selectedApplication === 'neural-network-playground' && (
        <NeuralNetworkPlayground />
      )}

      {selectedApplication === 'linear-regression' && (
        <LinearRegressionVisualization />
      )}

      {selectedApplication === 'pca-visualization' && (
        <PCAVisualization />
      )}

      {selectedApplication === 'convolution-visualization' && (
        <ConvolutionVisualization />
      )}

      {selectedApplication === 'k-means-clustering' && (
        <KMeansClusteringVisualization />
      )}

      {selectedApplication === 'trading-tools' && (
        <TradingToolsTutorial />
      )}
    </div>
  );
}

