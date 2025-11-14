import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, RotateCw, ArrowRight, ArrowDown } from 'lucide-react';

export default function InteractiveBackpropagationVisualization() {
  const canvasRef = useRef(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [phase, setPhase] = useState('forward'); // 'forward' or 'backward'

  // Simple 2-layer network: input -> hidden -> output
  const [forwardValues, setForwardValues] = useState({
    input: [1.0],
    hidden: [0.0, 0.0],
    output: 0.0,
    loss: 0.0
  });

  const [backwardGradients, setBackwardGradients] = useState({
    output: 0.0,
    hidden: [0.0, 0.0],
    weights2: [[0.0], [0.0]],
    weights1: [[0.0], [0.0]]
  });

  // Sample weights (simplified)
  const weights1 = [[0.5, -0.3], [0.2, 0.8]]; // 2x2: input -> hidden
  const weights2 = [[0.6, -0.4]]; // 1x2: hidden -> output
  const target = 0.8;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = rect.width > 0 ? rect.width : 800;
    const displayHeight = rect.height > 0 ? rect.height : 500;

    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    const width = displayWidth;
    const height = displayHeight;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw neural network structure
    const layerSpacing = width / 4;
    const nodeRadius = 30;
    const nodeSpacing = 80;

    // Input layer
    const inputX = layerSpacing;
    const inputY = height / 2;
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(inputX, inputY, nodeRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Input', inputX, inputY - nodeRadius - 10);
    ctx.fillText(forwardValues.input[0].toFixed(2), inputX, inputY + 5);

    // Hidden layer
    const hiddenX = layerSpacing * 2;
    const hiddenY1 = height / 2 - nodeSpacing / 2;
    const hiddenY2 = height / 2 + nodeSpacing / 2;

    [hiddenY1, hiddenY2].forEach((y, i) => {
      const isActive = phase === 'forward' && animationStep >= 2 + i;
      const hasGradient = phase === 'backward' && animationStep >= 5 + i;
      
      ctx.fillStyle = isActive ? '#10b981' : hasGradient ? '#f59e0b' : '#e5e7eb';
      ctx.beginPath();
      ctx.arc(hiddenX, y, nodeRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = isActive ? '#059669' : hasGradient ? '#d97706' : '#9ca3af';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`H${i + 1}`, hiddenX, y - nodeRadius - 10);
      ctx.fillText(forwardValues.hidden[i].toFixed(2), hiddenX, y + 5);
      if (hasGradient) {
        ctx.fillStyle = '#f59e0b';
        ctx.font = '10px sans-serif';
        ctx.fillText(`δ${i + 1}`, hiddenX, y + nodeRadius + 15);
      }
    });

    // Output layer
    const outputX = layerSpacing * 3;
    const outputY = height / 2;
    const isOutputActive = phase === 'forward' && animationStep >= 4;
    const hasOutputGradient = phase === 'backward' && animationStep >= 4;

    ctx.fillStyle = isOutputActive ? '#10b981' : hasOutputGradient ? '#ef4444' : '#e5e7eb';
    ctx.beginPath();
    ctx.arc(outputX, outputY, nodeRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = isOutputActive ? '#059669' : hasOutputGradient ? '#dc2626' : '#9ca3af';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Output', outputX, outputY - nodeRadius - 10);
    ctx.fillText(forwardValues.output.toFixed(2), outputX, outputY + 5);
    if (hasOutputGradient) {
      ctx.fillStyle = '#ef4444';
      ctx.font = '10px sans-serif';
      ctx.fillText(`∂L/∂ŷ`, outputX, outputY + nodeRadius + 15);
    }

    // Draw connections (forward pass)
    if (phase === 'forward' || animationStep < 5) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);

      // Input to Hidden
      [hiddenY1, hiddenY2].forEach((y, i) => {
        if (animationStep >= 1 + i) {
          ctx.beginPath();
          ctx.moveTo(inputX + nodeRadius, inputY);
          ctx.lineTo(hiddenX - nodeRadius, y);
          ctx.stroke();
        }
      });

      // Hidden to Output
      if (animationStep >= 3) {
        [hiddenY1, hiddenY2].forEach((y) => {
          ctx.beginPath();
          ctx.moveTo(hiddenX + nodeRadius, y);
          ctx.lineTo(outputX - nodeRadius, outputY);
          ctx.stroke();
        });
      }
    }

    // Draw connections (backward pass)
    if (phase === 'backward' && animationStep >= 4) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);

      // Output to Hidden
      if (animationStep >= 5) {
        [hiddenY1, hiddenY2].forEach((y) => {
          ctx.beginPath();
          ctx.moveTo(outputX - nodeRadius, outputY);
          ctx.lineTo(hiddenX + nodeRadius, y);
          ctx.stroke();
        });
      }

      // Hidden to Input
      if (animationStep >= 7) {
        [hiddenY1, hiddenY2].forEach((y, i) => {
          ctx.beginPath();
          ctx.moveTo(hiddenX - nodeRadius, y);
          ctx.lineTo(inputX + nodeRadius, inputY);
          ctx.stroke();
        });
      }

      ctx.setLineDash([]);
    }

    // Draw loss
    if (phase === 'forward' && animationStep >= 5) {
      ctx.fillStyle = '#f59e0b';
      ctx.font = 'bold 16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Loss: ${forwardValues.loss.toFixed(4)}`, width / 2, height - 30);
      ctx.fillText(`Target: ${target.toFixed(2)}`, width / 2, height - 10);
    }

    // Draw phase indicator
    ctx.fillStyle = phase === 'forward' ? '#3b82f6' : '#ef4444';
    ctx.font = 'bold 18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      phase === 'forward' ? '→ Forward Pass →' : '← Backward Pass ←',
      width / 2,
      30
    );
  }, [isAnimating, animationStep, phase, forwardValues, backwardGradients]);

  useEffect(() => {
    if (isAnimating) {
      let animationFrameId;
      let lastTime = 0;

      const animate = (currentTime) => {
        if (currentTime - lastTime >= 800) {
          setAnimationStep((prev) => {
            const next = prev + 1;

            // Forward pass: steps 0-5
            if (next <= 5 && phase === 'forward') {
              // Step 1-2: Compute hidden layer
              if (next === 2) {
                const h1 = forwardValues.input[0] * weights1[0][0];
                const h2 = forwardValues.input[0] * weights1[1][0];
                setForwardValues((prev) => ({
                  ...prev,
                  hidden: [h1, h2]
                }));
              }
              // Step 4: Compute output
              if (next === 4) {
                const output =
                  forwardValues.hidden[0] * weights2[0][0] +
                  forwardValues.hidden[1] * weights2[0][1];
                setForwardValues((prev) => ({
                  ...prev,
                  output
                }));
              }
              // Step 5: Compute loss
              if (next === 5) {
                const loss = Math.pow(forwardValues.output - target, 2);
                setForwardValues((prev) => ({
                  ...prev,
                  loss
                }));
                setIsAnimating(false);
                setTimeout(() => {
                  setPhase('backward');
                  setAnimationStep(0);
                }, 1000);
              }
            }

            // Backward pass: steps 0-8
            if (phase === 'backward') {
              // Step 4: Output gradient
              if (next === 4) {
                const outputGrad = 2 * (forwardValues.output - target);
                setBackwardGradients((prev) => ({
                  ...prev,
                  output: outputGrad
                }));
              }
              // Step 5-6: Hidden gradients
              if (next >= 5 && next <= 6) {
                const hiddenGrads = [
                  backwardGradients.output * weights2[0][0],
                  backwardGradients.output * weights2[0][1]
                ];
                setBackwardGradients((prev) => ({
                  ...prev,
                  hidden: hiddenGrads
                }));
              }
              // Step 8: Complete
              if (next >= 8) {
                setIsAnimating(false);
              }
            }

            return next;
          });
          lastTime = currentTime;
        }
        animationFrameId = requestAnimationFrame(animate);
      };

      animationFrameId = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationFrameId);
    }
  }, [isAnimating, phase, forwardValues, backwardGradients, target]);

  const startAnimation = () => {
    setPhase('forward');
    setAnimationStep(0);
    setForwardValues({
      input: [1.0],
      hidden: [0.0, 0.0],
      output: 0.0,
      loss: 0.0
    });
    setBackwardGradients({
      output: 0.0,
      hidden: [0.0, 0.0],
      weights2: [[0.0], [0.0]],
      weights1: [[0.0], [0.0]]
    });
    setIsAnimating(true);
  };

  const reset = () => {
    setIsAnimating(false);
    setAnimationStep(0);
    setPhase('forward');
    setForwardValues({
      input: [1.0],
      hidden: [0.0, 0.0],
      output: 0.0,
      loss: 0.0
    });
    setBackwardGradients({
      output: 0.0,
      hidden: [0.0, 0.0],
      weights2: [[0.0], [0.0]],
      weights1: [[0.0], [0.0]]
    });
  };

  return (
    <div className="space-y-4">
      <div className="bg-gray-50 rounded-lg p-4 mb-4">
        <div className="font-mono text-sm space-y-2">
          <div className="font-bold text-purple-700">Forward Pass:</div>
          <div>z₁ = W₁ × x + b₁</div>
          <div>a₁ = σ(z₁)</div>
          <div>ŷ = W₂ × a₁ + b₂</div>
          <div className="mt-2 pt-2 border-t border-gray-300">
            <div className="font-bold text-red-700">Backward Pass:</div>
            <div>∂L/∂ŷ = 2(ŷ - y)</div>
            <div>δ₂ = ∂L/∂ŷ × σ'(z₂)</div>
            <div>∂L/∂W₂ = δ₂ × a₁^T</div>
            <div>δ₁ = δ₂ × W₂^T × σ'(z₁)</div>
            <div>∂L/∂W₁ = δ₁ × x^T</div>
          </div>
        </div>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={startAnimation}
          disabled={isAnimating}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          Start Animation
        </button>
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          disabled={animationStep === 0}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isAnimating ? 'Pause' : 'Resume'}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
        >
          <RotateCw className="w-4 h-4" />
          Reset
        </button>
      </div>

      <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-blue-800">
          💡 <strong>Interactive:</strong> Watch the forward pass compute predictions, then the backward pass propagates gradients!
        </p>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />
    </div>
  );
}

