import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCw } from 'lucide-react';

export default function InteractiveEigenvalueVisualization({ matrix, eigenDecomp }) {
  const [selectedEigenvalue, setSelectedEigenvalue] = useState(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const canvasRef = useRef(null);

  const drawArrowhead = (ctx, x, y, angle, color) => {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-12, -6);
    ctx.lineTo(-12, 6);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  };

  useEffect(() => {
    // Use requestAnimationFrame to ensure canvas is ready
    const renderFrame = requestAnimationFrame(() => {
      if (!canvasRef.current || !eigenDecomp || selectedEigenvalue === null) {
        // Clear canvas if no selection
        if (canvasRef.current) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          if (canvas.width > 0 && canvas.height > 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Fill with white
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
          }
        }
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Handle high DPI displays
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      
      // Use default size if rect is not available yet
      const displayWidth = rect.width > 0 ? rect.width : 600;
      const displayHeight = rect.height > 0 ? rect.height : 400;
      
      // Only proceed if we have valid dimensions
      if (displayWidth <= 0 || displayHeight <= 0) return;
      
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
      ctx.scale(dpr, dpr);
      
      const width = displayWidth;
      const height = displayHeight;
      const centerX = width / 2;
      const centerY = height / 2;
      const gridScale = Math.min(width, height) / 10;

      // Clear canvas with white background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        const pos = (i - 5) * gridScale;
        ctx.beginPath();
        ctx.moveTo(centerX + pos, 0);
        ctx.lineTo(centerX + pos, height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, centerY + pos);
        ctx.lineTo(width, centerY + pos);
        ctx.stroke();
      }

      // Draw axes
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Draw axis labels
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.fillText('x', width - 20, centerY - 10);
      ctx.fillText('y', centerX + 10, 20);

      if (selectedEigenvalue !== null && 
          eigenDecomp.eigenvectors && 
          eigenDecomp.eigenvectors.length > selectedEigenvalue &&
          eigenDecomp.eigenvalues &&
          eigenDecomp.eigenvalues.length > selectedEigenvalue) {
        const eigenvectorObj = eigenDecomp.eigenvectors[selectedEigenvalue];
        const eigenvalue = eigenDecomp.eigenvalues[selectedEigenvalue];
        
        // Check if eigenvalue is real (same check as button selection)
        const isRealEigenvalue = typeof eigenvalue !== 'object' || 
                                 eigenvalue.imag === undefined || 
                                 Math.abs(eigenvalue.imag) < 1e-10;
        
        // Verify eigenvector object structure
        if (eigenvectorObj && 
            eigenvectorObj.eigenvector && 
            eigenvectorObj.eigenvector.x !== undefined &&
            eigenvectorObj.eigenvector.y !== undefined &&
            isRealEigenvalue) {
          const realEigenvalue = typeof eigenvalue === 'object' ? eigenvalue.real : eigenvalue;
          const eigenvector = eigenvectorObj.eigenvector;
          const vx = eigenvector.x || 0;
          const vy = eigenvector.y || 0;
          
          // Check if eigenvector is valid (not zero vector)
          if (Math.abs(vx) < 1e-10 && Math.abs(vy) < 1e-10) {
            // Use a default direction if eigenvector is zero
            const defaultVx = 1;
            const defaultVy = 0;
            const displayScale = Math.min(width, height) * 0.15;
            const displayVx = defaultVx * displayScale;
            const displayVy = defaultVy * displayScale;
            
            // Draw message
            ctx.fillStyle = '#ff6b6b';
            ctx.font = '12px sans-serif';
            ctx.fillText('Zero eigenvector - using default direction', centerX - width/2 + 20, height - 20);
            
            return; // Skip drawing for zero eigenvector
          }
          
          // Normalize eigenvector for better visualization if it's too small
          const magnitude = Math.sqrt(vx * vx + vy * vy);
          const normalizedVx = magnitude > 0 ? vx / magnitude : 1;
          const normalizedVy = magnitude > 0 ? vy / magnitude : 0;
          
          // Use normalized vector scaled appropriately
          const displayScale = Math.min(width, height) * 0.15;
          const displayVx = normalizedVx * displayScale;
          const displayVy = normalizedVy * displayScale;
          
          // Calculate transformed vector
          const t = animationStep / 60; // 0 to 1
          const transformedX = matrix[0][0] * displayVx + matrix[0][1] * displayVy;
          const transformedY = matrix[1][0] * displayVx + matrix[1][1] * displayVy;
          
          // Interpolate for animation
          const currentX = displayVx + (transformedX - displayVx) * t;
          const currentY = displayVy + (transformedY - displayVy) * t;
          
          // Draw original eigenvector
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(centerX + displayVx, centerY - displayVy);
          ctx.stroke();
          
          // Draw arrowhead for original
          const angle = Math.atan2(-displayVy, displayVx);
          drawArrowhead(ctx, centerX + displayVx, centerY - displayVy, angle, '#3b82f6');
          
          // Draw transformed vector
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(centerX + transformedX, centerY - transformedY);
          ctx.stroke();
          ctx.setLineDash([]);
          
          // Draw arrowhead for transformed
          const angle2 = Math.atan2(-transformedY, transformedX);
          drawArrowhead(ctx, centerX + transformedX, centerY - transformedY, angle2, '#10b981');
          
          // Draw animated vector
          if (isAnimating && animationStep > 0) {
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(centerX + currentX, centerY - currentY);
            ctx.stroke();
            
            const angle3 = Math.atan2(-currentY, currentX);
            drawArrowhead(ctx, centerX + currentX, centerY - currentY, angle3, '#f59e0b');
          }
          
          // Draw labels
          ctx.fillStyle = '#3b82f6';
          ctx.font = 'bold 14px sans-serif';
          ctx.fillText('v (eigenvector)', centerX + displayVx + 15, centerY - displayVy - 10);
          
          ctx.fillStyle = '#10b981';
          ctx.fillText('A×v', centerX + transformedX + 15, centerY - transformedY - 10);
          
          ctx.fillStyle = '#6b7280';
          ctx.font = '12px sans-serif';
          ctx.fillText(`λ = ${realEigenvalue.toFixed(3)}`, centerX - width/2 + 20, 30);
          ctx.fillText(`A×v = ${realEigenvalue.toFixed(3)}×v`, centerX - width/2 + 20, 50);
        }
      }
    });

    return () => cancelAnimationFrame(renderFrame);
  }, [matrix, eigenDecomp, selectedEigenvalue, isAnimating, animationStep]);

  useEffect(() => {
    if (isAnimating) {
      let animationFrameId;
      let lastTime = 0;
      
      const animate = (currentTime) => {
        if (currentTime - lastTime >= 16) { // ~60fps
          setAnimationStep(prev => (prev + 1) % 60);
          lastTime = currentTime;
        }
        animationFrameId = requestAnimationFrame(animate);
      };
      
      animationFrameId = requestAnimationFrame(animate);
      return () => cancelAnimationFrame(animationFrameId);
    } else {
      // Keep rendering even when not animating
      setAnimationStep(0);
    }
  }, [isAnimating]);

  // Handle window resize - trigger re-render
  useEffect(() => {
    const handleResize = () => {
      // Small delay to ensure canvas has new dimensions
      setTimeout(() => {
        if (canvasRef.current && selectedEigenvalue !== null) {
          // Force re-render by toggling animation step
          setAnimationStep(prev => prev === 0 ? 0.1 : 0);
        }
      }, 100);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [selectedEigenvalue]);

  if (!eigenDecomp || !eigenDecomp.eigenvalues) return null;

  return (
    <div className="space-y-4">
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border-2 border-blue-200">
        <h4 className="font-bold text-blue-900 mb-3">Interactive Eigenvector Visualization</h4>
        
        {/* Eigenvalue Selection */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Select Eigenvalue to Visualize:
          </label>
          <div className="flex gap-2 flex-wrap">
            {Array.isArray(eigenDecomp.eigenvalues) ? (
              eigenDecomp.eigenvalues.map((ev, i) => {
                // Check if eigenvalue is real (matches rendering logic)
                const isReal = typeof ev !== 'object' || ev.imag === undefined || Math.abs(ev.imag) < 1e-10;
                // Check if corresponding eigenvector exists
                const hasEigenvector = eigenDecomp.eigenvectors && 
                                      eigenDecomp.eigenvectors[i] && 
                                      eigenDecomp.eigenvectors[i].eigenvector;
                const canVisualize = isReal && hasEigenvector;
                
                return (
                  <button
                    key={i}
                    onClick={() => {
                      if (canVisualize) {
                        setSelectedEigenvalue(i);
                        setIsAnimating(false);
                        setAnimationStep(0);
                      }
                    }}
                    disabled={!canVisualize}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      selectedEigenvalue === i
                        ? 'bg-blue-600 text-white ring-2 ring-blue-800'
                        : canVisualize
                        ? 'bg-white text-blue-600 border-2 border-blue-300 hover:bg-blue-50'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    λ{i + 1} = {typeof ev === 'object' && ev.imag !== undefined && Math.abs(ev.imag) >= 1e-10
                      ? `${ev.real.toFixed(2)}${ev.imag >= 0 ? '+' : ''}${ev.imag.toFixed(2)}i`
                      : (typeof ev === 'object' ? ev.real.toFixed(2) : ev.toFixed(2))}
                    {!canVisualize && <span className="ml-2 text-xs">(no eigenvector)</span>}
                  </button>
                );
              })
            ) : null}
          </div>
        </div>

        {/* Animation Controls */}
        {selectedEigenvalue !== null && (
          <div className="flex items-center gap-3 mb-4">
            <button
              onClick={() => setIsAnimating(!isAnimating)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              <span>{isAnimating ? 'Pause' : 'Play'} Animation</span>
            </button>
            <button
              onClick={() => {
                setAnimationStep(0);
                setIsAnimating(false);
              }}
              className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
            >
              <RotateCw className="w-4 h-4" />
              <span>Reset</span>
            </button>
          </div>
        )}

        {/* Canvas Visualization */}
        {selectedEigenvalue !== null && (
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
            <canvas
              ref={canvasRef}
              className="w-full border border-gray-300 rounded-lg"
              style={{ height: '400px', width: '100%' }}
            />
            {/* Debug info */}
            {process.env.NODE_ENV === 'development' && (
              <div className="mt-2 text-xs text-gray-500">
                Debug: Selected index={selectedEigenvalue}, 
                Eigenvalue={eigenDecomp.eigenvalues[selectedEigenvalue] ? 
                  (typeof eigenDecomp.eigenvalues[selectedEigenvalue] === 'object' ? 
                    `${eigenDecomp.eigenvalues[selectedEigenvalue].real.toFixed(3)}` : 
                    eigenDecomp.eigenvalues[selectedEigenvalue].toFixed(3)) : 'N/A'}, 
                Has eigenvector={eigenDecomp.eigenvectors && eigenDecomp.eigenvectors[selectedEigenvalue] ? 'Yes' : 'No'}
              </div>
            )}
            <div className="mt-3 flex items-center justify-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-blue-600"></div>
                <span>Original Eigenvector</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-green-500 border-dashed border-2"></div>
                <span>Transformed Vector (A×v)</span>
              </div>
              {isAnimating && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-orange-500"></div>
                  <span>Animation</span>
                </div>
              )}
            </div>
          </div>
        )}

        {selectedEigenvalue === null && (
          <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4 text-center">
            <p className="text-yellow-900 text-sm">
              💡 Select an eigenvalue above to see how the eigenvector transforms
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

