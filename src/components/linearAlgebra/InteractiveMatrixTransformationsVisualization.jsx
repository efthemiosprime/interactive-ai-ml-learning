import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import * as transforms from '../../utils/matrixTransformations';

export default function InteractiveMatrixTransformationsVisualization() {
  const canvasRef = useRef(null);
  const [transformationType, setTransformationType] = useState('translation');
  const [dimension, setDimension] = useState('2d');
  
  // 2D parameters
  const [translationX, setTranslationX] = useState(1);
  const [translationY, setTranslationY] = useState(0.5);
  const [rotationAngle, setRotationAngle] = useState(45);
  const [scaleX, setScaleX] = useState(1.5);
  const [scaleY, setScaleY] = useState(1.5);
  const [reflectionAxis, setReflectionAxis] = useState('x');
  
  // 3D parameters
  const [rotation3DX, setRotation3DX] = useState(30);
  const [rotation3DY, setRotation3DY] = useState(30);
  const [rotation3DZ, setRotation3DZ] = useState(0);
  const [scale3D, setScale3D] = useState(1.2);
  const [showCalculations, setShowCalculations] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationProgress, setAnimationProgress] = useState(0);
  
  // Shear parameters
  const [shearX, setShearX] = useState(0.5);
  const [shearY, setShearY] = useState(0);
  
  // Combined transformations
  const [enableCombined, setEnableCombined] = useState(false);
  const [transformation1, setTransformation1] = useState('translation');
  const [transformation2, setTransformation2] = useState('rotation');

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
    const padding = 60;
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = 50; // Pixels per unit

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (dimension === '2d') {
      // 2D Transformations
      const originalShape = transforms.generateSquare2D(2, 0, 0);
      
      // Apply transformation
      let transformationMatrix;
      switch (transformationType) {
        case 'translation':
          transformationMatrix = transforms.translation2D(translationX, translationY);
          break;
        case 'rotation':
          transformationMatrix = transforms.rotation2D((rotationAngle * Math.PI) / 180);
          break;
        case 'scaling':
          transformationMatrix = transforms.scaling2D(scaleX, scaleY);
          break;
        case 'reflection':
          transformationMatrix = transforms.reflection2D(reflectionAxis);
          break;
        case 'shear':
          transformationMatrix = transforms.shear2D(shearX, shearY);
          break;
        default:
          transformationMatrix = transforms.translation2D(0, 0);
      }

      // Apply combined transformations if enabled
      if (enableCombined) {
        let matrix1, matrix2;
        
        // Build first transformation
        switch (transformation1) {
          case 'translation':
            matrix1 = transforms.translation2D(translationX, translationY);
            break;
          case 'rotation':
            matrix1 = transforms.rotation2D((rotationAngle * Math.PI) / 180);
            break;
          case 'scaling':
            matrix1 = transforms.scaling2D(scaleX, scaleY);
            break;
          case 'reflection':
            matrix1 = transforms.reflection2D(reflectionAxis);
            break;
          case 'shear':
            matrix1 = transforms.shear2D(shearX, shearY);
            break;
          default:
            matrix1 = transforms.translation2D(0, 0);
        }
        
        // Build second transformation
        switch (transformation2) {
          case 'translation':
            matrix2 = transforms.translation2D(translationX, translationY);
            break;
          case 'rotation':
            matrix2 = transforms.rotation2D((rotationAngle * Math.PI) / 180);
            break;
          case 'scaling':
            matrix2 = transforms.scaling2D(scaleX, scaleY);
            break;
          case 'reflection':
            matrix2 = transforms.reflection2D(reflectionAxis);
            break;
          case 'shear':
            matrix2 = transforms.shear2D(shearX, shearY);
            break;
          default:
            matrix2 = transforms.translation2D(0, 0);
        }
        
        // Combine: T2 × T1 (apply T1 first, then T2)
        transformationMatrix = transforms.multiplyTransformMatrices(matrix2, matrix1);
      }

      // Interpolate for animation
      let transformedShape;
      if (isAnimating && animationProgress < 1) {
        transformedShape = originalShape.map((point, idx) => {
          const finalPoint = transforms.transformPoint2D(point, transformationMatrix);
          return [
            point[0] + (finalPoint[0] - point[0]) * animationProgress,
            point[1] + (finalPoint[1] - point[1]) * animationProgress
          ];
        });
      } else {
        transformedShape = transforms.transformPoints2D(originalShape, transformationMatrix);
      }

      // Draw axes
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, centerY);
      ctx.lineTo(width - padding, centerY);
      ctx.moveTo(centerX, padding);
      ctx.lineTo(centerX, height - padding);
      ctx.stroke();

      // Draw original shape
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      originalShape.forEach((point, idx) => {
        const x = centerX + point[0] * scale;
        const y = centerY - point[1] * scale;
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw transformed shape
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      transformedShape.forEach((point, idx) => {
        const x = centerX + point[0] * scale;
        const y = centerY - point[1] * scale;
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw arrows showing transformation direction
      // Show arrows when not animating or when animation is past 30%
      if (!isAnimating || animationProgress > 0.3) {
        originalShape.forEach((origPoint, idx) => {
          const finalTransformedPoint = transforms.transformPoint2D(origPoint, transformationMatrix);
          const origX = centerX + origPoint[0] * scale;
          const origY = centerY - origPoint[1] * scale;
          
          // Use current transformed point for arrow end during animation
          const transX = centerX + transformedShape[idx][0] * scale;
          const transY = centerY - transformedShape[idx][1] * scale;
          
          const dx = transX - origX;
          const dy = transY - origY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          if (dist > 2) {
            // Draw arrow with opacity based on animation progress
            const opacity = isAnimating ? Math.min(animationProgress / 0.3, 1) : 1;
            ctx.strokeStyle = `rgba(245, 158, 11, ${opacity})`;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(origX, origY);
            ctx.lineTo(transX, transY);
            ctx.stroke();
            
            // Arrowhead
            const angle = Math.atan2(dy, dx);
            const arrowLength = 8;
            const arrowAngle = Math.PI / 6;
            ctx.beginPath();
            ctx.moveTo(transX, transY);
            ctx.lineTo(
              transX - arrowLength * Math.cos(angle - arrowAngle),
              transY - arrowLength * Math.sin(angle - arrowAngle)
            );
            ctx.moveTo(transX, transY);
            ctx.lineTo(
              transX - arrowLength * Math.cos(angle + arrowAngle),
              transY - arrowLength * Math.sin(angle + arrowAngle)
            );
            ctx.stroke();
          }
        });
      }

      // Draw labels
      ctx.fillStyle = '#3b82f6';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('Original', centerX - 60, centerY - 30);
      ctx.fillStyle = '#10b981';
      ctx.fillText('Transformed', centerX + 20, centerY - 30);
    } else {
      // 3D Transformations
      const originalCube = transforms.generateCube3D(2, 0, 0, 0);
      
      // Apply transformations
      let transformationMatrix = transforms.scaling3D(scale3D, scale3D, scale3D);
      transformationMatrix = transforms.multiplyTransformMatrices(
        transforms.rotation3DX((rotation3DX * Math.PI) / 180),
        transformationMatrix
      );
      transformationMatrix = transforms.multiplyTransformMatrices(
        transforms.rotation3DY((rotation3DY * Math.PI) / 180),
        transformationMatrix
      );
      transformationMatrix = transforms.multiplyTransformMatrices(
        transforms.rotation3DZ((rotation3DZ * Math.PI) / 180),
        transformationMatrix
      );

      const transformedCube = originalCube.map(point => 
        transforms.transformPoint3D(point, transformationMatrix)
      );

      // Project to 2D
      const projectedOriginal = originalCube.map(point => 
        transforms.project3DTo2D(point, 5)
      );
      const projectedTransformed = transformedCube.map(point => 
        transforms.project3DTo2D(point, 5)
      );

      // Draw axes
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, centerY);
      ctx.lineTo(width - padding, centerY);
      ctx.moveTo(centerX, padding);
      ctx.lineTo(centerX, height - padding);
      ctx.stroke();

      // Draw original cube
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      projectedOriginal.forEach((point, idx) => {
        const x = centerX + point[0] * scale;
        const y = centerY - point[1] * scale;
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw transformed cube
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      projectedTransformed.forEach((point, idx) => {
        const x = centerX + point[0] * scale;
        const y = centerY - point[1] * scale;
        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw title
    ctx.fillStyle = '#1f2937';
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    const titles = {
      translation: enableCombined ? 'Combined Transformations' : 'Translation',
      rotation: enableCombined ? 'Combined Transformations' : 'Rotation',
      scaling: enableCombined ? 'Combined Transformations' : 'Scaling',
      reflection: enableCombined ? 'Combined Transformations' : 'Reflection',
      shear: enableCombined ? 'Combined Transformations' : 'Shear'
    };
    ctx.fillText(titles[transformationType] || 'Transformation', width / 2, 25);
  }, [
    transformationType, dimension, translationX, translationY, rotationAngle,
    scaleX, scaleY, reflectionAxis, rotation3DX, rotation3DY, rotation3DZ, scale3D,
    isAnimating, animationProgress, shearX, shearY, enableCombined, transformation1, transformation2
  ]);

  // Animation effect
  useEffect(() => {
    if (isAnimating) {
      const duration = 1000; // 1 second
      const startTime = Date.now();
      let animationFrameId;
      
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        setAnimationProgress(progress);
        
        if (progress < 1) {
          animationFrameId = requestAnimationFrame(animate);
        } else {
          // Animation complete - keep final state visible for a moment
          setTimeout(() => {
            setIsAnimating(false);
            setAnimationProgress(0);
          }, 500);
        }
      };
      
      animationFrameId = requestAnimationFrame(animate);
      
      return () => {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
      };
    }
  }, [isAnimating]);

  return (
    <div className="space-y-4">
      <div className="bg-indigo-50 border-2 border-indigo-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-indigo-800">
          💡 <strong>Interactive:</strong> See how matrices transform shapes in 2D and 3D space!
        </p>
      </div>

      {/* Operation Info */}
      <div className="bg-gradient-to-r from-indigo-100 to-purple-100 rounded-lg p-4 border-2 border-indigo-300">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-indigo-900 mb-1">
              {transformationType === 'translation' && (enableCombined ? 'Combined Transformations' : 'Translation Operation')}
              {transformationType === 'rotation' && (enableCombined ? 'Combined Transformations' : 'Rotation Operation')}
              {transformationType === 'scaling' && (enableCombined ? 'Combined Transformations' : 'Scaling Operation')}
              {transformationType === 'reflection' && (enableCombined ? 'Combined Transformations' : 'Reflection Operation')}
              {transformationType === 'shear' && (enableCombined ? 'Combined Transformations' : 'Shear Operation')}
            </h3>
            <p className="text-sm text-indigo-800">
              {enableCombined && `Applying ${transformation1} then ${transformation2} (T2 × T1)`}
              {!enableCombined && transformationType === 'translation' && `Moving shape by (${translationX.toFixed(2)}, ${translationY.toFixed(2)})`}
              {!enableCombined && transformationType === 'rotation' && `Rotating shape by ${rotationAngle}° around origin`}
              {!enableCombined && transformationType === 'scaling' && `Scaling shape by ${scaleX.toFixed(2)}× horizontally, ${scaleY.toFixed(2)}× vertically`}
              {!enableCombined && transformationType === 'reflection' && `Reflecting shape across ${reflectionAxis === 'x' ? 'X-axis' : reflectionAxis === 'y' ? 'Y-axis' : 'origin'}`}
              {!enableCombined && transformationType === 'shear' && `Shearing shape by (${shearX.toFixed(2)}, ${shearY.toFixed(2)})`}
            </p>
          </div>
          <button
            onClick={() => {
              if (!isAnimating) {
                setAnimationProgress(0);
                setIsAnimating(true);
              }
            }}
            disabled={isAnimating}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-all"
          >
            {isAnimating ? 'Animating...' : '▶ Demo Operation'}
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg p-4 border-2 border-indigo-200 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Dimension
          </label>
          <select
            value={dimension}
            onChange={(e) => {
              setDimension(e.target.value);
              setIsAnimating(false);
              setAnimationProgress(0);
            }}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
          >
            <option value="2d">2D</option>
            <option value="3d">3D</option>
          </select>
        </div>

        {dimension === '2d' && (
          <>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Transformation Type
              </label>
              <select
                value={transformationType}
                onChange={(e) => {
                  setTransformationType(e.target.value);
                  setIsAnimating(false);
                  setAnimationProgress(0);
                }}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              >
                <option value="translation">Translation</option>
                <option value="rotation">Rotation</option>
                <option value="scaling">Scaling</option>
                <option value="reflection">Reflection</option>
                <option value="shear">Shear</option>
              </select>
            </div>

            {transformationType === 'translation' && (
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-600">Translation X: {translationX.toFixed(2)}</label>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={translationX}
                    onChange={(e) => {
                      setTranslationX(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Translation Y: {translationY.toFixed(2)}</label>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={translationY}
                    onChange={(e) => {
                      setTranslationY(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {transformationType === 'rotation' && (
              <div>
                <label className="text-xs text-gray-600">Angle: {rotationAngle}°</label>
                <input
                  type="range"
                  min="0"
                  max="360"
                  step="5"
                    value={rotationAngle}
                    onChange={(e) => {
                      setRotationAngle(parseInt(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
              </div>
            )}

            {transformationType === 'scaling' && (
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-600">Scale X: {scaleX.toFixed(2)}</label>
                  <input
                    type="range"
                    min="0.5"
                    max="3"
                    step="0.1"
                    value={scaleX}
                    onChange={(e) => {
                      setScaleX(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Scale Y: {scaleY.toFixed(2)}</label>
                  <input
                    type="range"
                    min="0.5"
                    max="3"
                    step="0.1"
                    value={scaleY}
                    onChange={(e) => {
                      setScaleY(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {transformationType === 'reflection' && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Reflection Axis
                </label>
                <select
                  value={reflectionAxis}
                  onChange={(e) => {
                    setReflectionAxis(e.target.value);
                    setIsAnimating(false);
                    setAnimationProgress(0);
                  }}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="x">X-axis</option>
                  <option value="y">Y-axis</option>
                  <option value="origin">Origin</option>
                </select>
              </div>
            )}

            {transformationType === 'shear' && (
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-gray-600">Shear X: {shearX.toFixed(2)}</label>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={shearX}
                    onChange={(e) => {
                      setShearX(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-600">Shear Y: {shearY.toFixed(2)}</label>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={shearY}
                    onChange={(e) => {
                      setShearY(parseFloat(e.target.value));
                      setIsAnimating(false);
                      setAnimationProgress(0);
                    }}
                    className="w-full"
                  />
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  Shear distorts the shape by sliding along axes
                </div>
              </div>
            )}

            {/* Combined Transformations */}
            <div className="border-t pt-4 mt-4">
              <label className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  checked={enableCombined}
                  onChange={(e) => {
                    setEnableCombined(e.target.checked);
                    setIsAnimating(false);
                    setAnimationProgress(0);
                  }}
                  className="w-4 h-4"
                />
                <span className="text-sm font-semibold text-gray-700">
                  Enable Combined Transformations
                </span>
              </label>
              {enableCombined && (
                <div className="mt-3 space-y-3 bg-indigo-50 p-3 rounded-lg">
                  <div className="text-xs text-indigo-800 mb-2">
                    <strong>Note:</strong> Order matters! T2 × T1 means apply T1 first, then T2
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1">
                      First Transformation (T1)
                    </label>
                    <select
                      value={transformation1}
                      onChange={(e) => {
                        setTransformation1(e.target.value);
                        setIsAnimating(false);
                        setAnimationProgress(0);
                      }}
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="translation">Translation</option>
                      <option value="rotation">Rotation</option>
                      <option value="scaling">Scaling</option>
                      <option value="reflection">Reflection</option>
                      <option value="shear">Shear</option>
                    </select>
                  </div>
                  <div className="text-center text-lg font-bold text-indigo-600">×</div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1">
                      Second Transformation (T2)
                    </label>
                    <select
                      value={transformation2}
                      onChange={(e) => {
                        setTransformation2(e.target.value);
                        setIsAnimating(false);
                        setAnimationProgress(0);
                      }}
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="translation">Translation</option>
                      <option value="rotation">Rotation</option>
                      <option value="scaling">Scaling</option>
                      <option value="reflection">Reflection</option>
                      <option value="shear">Shear</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {dimension === '3d' && (
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-600">Rotation X: {rotation3DX}°</label>
              <input
                type="range"
                min="0"
                max="360"
                step="5"
                value={rotation3DX}
                onChange={(e) => setRotation3DX(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-xs text-gray-600">Rotation Y: {rotation3DY}°</label>
              <input
                type="range"
                min="0"
                max="360"
                step="5"
                value={rotation3DY}
                onChange={(e) => setRotation3DY(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-xs text-gray-600">Rotation Z: {rotation3DZ}°</label>
              <input
                type="range"
                min="0"
                max="360"
                step="5"
                value={rotation3DZ}
                onChange={(e) => setRotation3DZ(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="text-xs text-gray-600">Scale: {scale3D.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={scale3D}
                onChange={(e) => setScale3D(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="w-full border border-gray-300 rounded-lg"
        style={{ height: '500px' }}
      />

      {/* Matrix Display */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-900 mb-2">Transformation Matrix:</h4>
        <div className="font-mono text-sm bg-white p-3 rounded border space-y-3">
          {dimension === '2d' && transformationType === 'translation' && (
            <div>
              <div className="mb-2">
                <div>[1  0  {translationX.toFixed(2)}]</div>
                <div>[0  1  {translationY.toFixed(2)}]</div>
                <div>[0  0  1]</div>
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Calculation:</strong></div>
                <div>Translation X = {translationX.toFixed(2)}</div>
                <div>Translation Y = {translationY.toFixed(2)}</div>
                <div className="mt-1">Matrix form: [1  0  tx]</div>
                <div>                [0  1  ty]</div>
                <div>                [0  0   1]</div>
              </div>
            </div>
          )}
          {dimension === '2d' && transformationType === 'rotation' && (
            <div>
              <div className="mb-2">
                <div>[{Math.cos((rotationAngle * Math.PI) / 180).toFixed(3)}  {-Math.sin((rotationAngle * Math.PI) / 180).toFixed(3)}  0]</div>
                <div>[{Math.sin((rotationAngle * Math.PI) / 180).toFixed(3)}  {Math.cos((rotationAngle * Math.PI) / 180).toFixed(3)}  0]</div>
                <div>[0  0  1]</div>
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Calculation:</strong></div>
                <div>Angle = {rotationAngle}° = {(rotationAngle * Math.PI / 180).toFixed(4)} radians</div>
                <div>cos({rotationAngle}°) = {Math.cos((rotationAngle * Math.PI) / 180).toFixed(4)}</div>
                <div>sin({rotationAngle}°) = {Math.sin((rotationAngle * Math.PI) / 180).toFixed(4)}</div>
                <div className="mt-1">Matrix form: [cos(θ)  -sin(θ)  0]</div>
                <div>                [sin(θ)   cos(θ)  0]</div>
                <div>                [   0        0    1]</div>
              </div>
            </div>
          )}
          {dimension === '2d' && transformationType === 'scaling' && (
            <div>
              <div className="mb-2">
                <div>[{scaleX.toFixed(2)}  0  0]</div>
                <div>[0  {scaleY.toFixed(2)}  0]</div>
                <div>[0  0  1]</div>
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Calculation:</strong></div>
                <div>Scale X = {scaleX.toFixed(2)}</div>
                <div>Scale Y = {scaleY.toFixed(2)}</div>
                <div className="mt-1">Matrix form: [sx   0   0]</div>
                <div>                [ 0  sy   0]</div>
                <div>                [ 0   0   1]</div>
                <div className="mt-1 text-gray-500">
                  {scaleX === scaleY ? 'Uniform scaling' : 'Non-uniform scaling'}
                </div>
              </div>
            </div>
          )}
          {dimension === '2d' && transformationType === 'shear' && (
            <div>
              <div className="mb-2">
                <div>[1  {shearX.toFixed(2)}  0]</div>
                <div>[{shearY.toFixed(2)}  1  0]</div>
                <div>[0  0  1]</div>
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Calculation:</strong></div>
                <div>Shear X = {shearX.toFixed(2)}</div>
                <div>Shear Y = {shearY.toFixed(2)}</div>
                <div className="mt-1">Matrix form: [1  shx   0]</div>
                <div>                [shy   1   0]</div>
                <div>                [ 0    0   1]</div>
                <div className="mt-1 text-gray-500">
                  Shear distorts shapes by sliding along axes (like pushing a deck of cards)
                </div>
              </div>
            </div>
          )}
          {dimension === '2d' && transformationType === 'reflection' && (
            <div>
              <div className="mb-2">
                {reflectionAxis === 'x' && (
                  <>
                    <div>[1  0  0]</div>
                    <div>[0  -1  0]</div>
                    <div>[0  0  1]</div>
                  </>
                )}
                {reflectionAxis === 'y' && (
                  <>
                    <div>[-1  0  0]</div>
                    <div>[0  1  0]</div>
                    <div>[0  0  1]</div>
                  </>
                )}
                {reflectionAxis === 'origin' && (
                  <>
                    <div>[-1  0  0]</div>
                    <div>[0  -1  0]</div>
                    <div>[0  0  1]</div>
                  </>
                )}
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Calculation:</strong></div>
                <div>Reflection axis: {reflectionAxis === 'x' ? 'X-axis' : reflectionAxis === 'y' ? 'Y-axis' : 'Origin'}</div>
                {reflectionAxis === 'x' && (
                  <>
                    <div className="mt-1">Reflects across X-axis (y → -y)</div>
                    <div>Matrix: [1   0   0]</div>
                    <div>        [0  -1   0]</div>
                    <div>        [0   0   1]</div>
                  </>
                )}
                {reflectionAxis === 'y' && (
                  <>
                    <div className="mt-1">Reflects across Y-axis (x → -x)</div>
                    <div>Matrix: [-1   0   0]</div>
                    <div>        [ 0   1   0]</div>
                    <div>        [ 0   0   1]</div>
                  </>
                )}
                {reflectionAxis === 'origin' && (
                  <>
                    <div className="mt-1">Reflects across origin (x → -x, y → -y)</div>
                    <div>Matrix: [-1   0   0]</div>
                    <div>        [ 0  -1   0]</div>
                    <div>        [ 0   0   1]</div>
                  </>
                )}
              </div>
            </div>
          )}
          {dimension === '2d' && enableCombined && (
            <div>
              <div className="mb-2">
                {(() => {
                  let matrix1, matrix2;
                  switch (transformation1) {
                    case 'translation': matrix1 = transforms.translation2D(translationX, translationY); break;
                    case 'rotation': matrix1 = transforms.rotation2D((rotationAngle * Math.PI) / 180); break;
                    case 'scaling': matrix1 = transforms.scaling2D(scaleX, scaleY); break;
                    case 'reflection': matrix1 = transforms.reflection2D(reflectionAxis); break;
                    case 'shear': matrix1 = transforms.shear2D(shearX, shearY); break;
                    default: matrix1 = transforms.translation2D(0, 0);
                  }
                  switch (transformation2) {
                    case 'translation': matrix2 = transforms.translation2D(translationX, translationY); break;
                    case 'rotation': matrix2 = transforms.rotation2D((rotationAngle * Math.PI) / 180); break;
                    case 'scaling': matrix2 = transforms.scaling2D(scaleX, scaleY); break;
                    case 'reflection': matrix2 = transforms.reflection2D(reflectionAxis); break;
                    case 'shear': matrix2 = transforms.shear2D(shearX, shearY); break;
                    default: matrix2 = transforms.translation2D(0, 0);
                  }
                  const combined = transforms.multiplyTransformMatrices(matrix2, matrix1);
                  return (
                    <>
                      {combined.map((row, i) => (
                        <div key={i}>
                          [{row.map((val, j) => (
                            <span key={j} className={j < row.length - 1 ? 'mr-2' : ''}>
                              {val.toFixed(3)}
                            </span>
                          ))}]
                        </div>
                      ))}
                    </>
                  );
                })()}
              </div>
              <div className="text-xs text-gray-600 border-t pt-2 mt-2">
                <div><strong>Combined Matrix Calculation:</strong></div>
                <div className="mt-1">T_combined = T2 × T1</div>
                <div className="mt-1">Where:</div>
                <div>• T1 = {transformation1} (applied first)</div>
                <div>• T2 = {transformation2} (applied second)</div>
                <div className="mt-2 text-orange-600">
                  <strong>⚠️ Order matters!</strong> T2 × T1 ≠ T1 × T2 in general
                </div>
              </div>
            </div>
          )}
          {dimension === '3d' && (
            <div className="text-xs">
              <div>4×4 Transformation Matrix</div>
              <div className="text-gray-600 mt-1">(Combined rotation, scaling)</div>
              <div className="border-t pt-2 mt-2 text-gray-600">
                <div><strong>Operations:</strong></div>
                <div>Rotation X: {rotation3DX}°</div>
                <div>Rotation Y: {rotation3DY}°</div>
                <div>Rotation Z: {rotation3DZ}°</div>
                <div>Scale: {scale3D.toFixed(2)}×</div>
                <div className="mt-1">Matrix = Rz × Ry × Rx × S</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Step-by-Step Calculation */}
      {dimension === '2d' && (
        <div className="bg-indigo-50 rounded-lg p-4 border-2 border-indigo-200">
          <button
            onClick={() => setShowCalculations(!showCalculations)}
            className="w-full flex items-center justify-between mb-2"
          >
            <h4 className="font-semibold text-indigo-900">
              Step-by-Step Calculation
            </h4>
            {showCalculations ? (
              <ChevronUp className="w-5 h-5 text-indigo-600" />
            ) : (
              <ChevronDown className="w-5 h-5 text-indigo-600" />
            )}
          </button>

          {showCalculations && (
            <div className="space-y-4 mt-4">
              {/* Point Selector */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Select Point to Calculate:
                </label>
                <select
                  value={selectedPoint}
                  onChange={(e) => setSelectedPoint(parseInt(e.target.value))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  {(() => {
                    const square = transforms.generateSquare2D(2, 0, 0);
                    return square.slice(0, 4).map((_, idx) => (
                      <option key={idx} value={idx}>
                        Point {idx + 1}: ({square[idx][0].toFixed(1)}, {square[idx][1].toFixed(1)})
                      </option>
                    ));
                  })()}
                </select>
              </div>

              {/* Calculation Steps */}
              {(() => {
                const square = transforms.generateSquare2D(2, 0, 0);
                const originalPoint = square[selectedPoint];
                const originalHomogeneous = [originalPoint[0], originalPoint[1], 1];

                let transformationMatrix;
                switch (transformationType) {
                  case 'translation':
                    transformationMatrix = transforms.translation2D(translationX, translationY);
                    break;
                  case 'rotation':
                    transformationMatrix = transforms.rotation2D((rotationAngle * Math.PI) / 180);
                    break;
                  case 'scaling':
                    transformationMatrix = transforms.scaling2D(scaleX, scaleY);
                    break;
                  case 'reflection':
                    transformationMatrix = transforms.reflection2D(reflectionAxis);
                    break;
                  default:
                    transformationMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
                }

                const transformedPoint = transforms.transformPoint2D(originalPoint, transformationMatrix);

                return (
                  <div className="bg-white rounded-lg p-4 space-y-4">
                    {/* Step 1: Original Point */}
                    <div>
                      <h5 className="font-semibold text-gray-800 mb-2">Step 1: Original Point (Homogeneous Coordinates)</h5>
                      <div className="bg-blue-50 rounded p-3 border border-blue-200">
                        <div className="font-mono text-sm">
                          <div className="text-center mb-2">
                            <div className="inline-block border-2 border-blue-300 rounded p-2">
                              <div>[{originalHomogeneous[0].toFixed(2)}]</div>
                              <div>[{originalHomogeneous[1].toFixed(2)}]</div>
                              <div>[{originalHomogeneous[2]}]</div>
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 text-center">
                            Original: ({originalPoint[0].toFixed(2)}, {originalPoint[1].toFixed(2)}) → Homogeneous: [{originalHomogeneous[0].toFixed(2)}, {originalHomogeneous[1].toFixed(2)}, 1]
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Step 2: Transformation Matrix */}
                    <div>
                      <h5 className="font-semibold text-gray-800 mb-2">Step 2: Transformation Matrix</h5>
                      <div className="bg-purple-50 rounded p-3 border border-purple-200">
                        <div className="font-mono text-sm">
                          <div className="text-center mb-2">
                            <div className="inline-block border-2 border-purple-300 rounded p-2">
                              {transformationMatrix.map((row, i) => (
                                <div key={i}>
                                  [{row.map((val, j) => (
                                    <span key={j} className={j < row.length - 1 ? 'mr-2' : ''}>
                                      {val.toFixed(3)}
                                    </span>
                                  ))}]
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Step 3: Matrix Multiplication */}
                    <div>
                      <h5 className="font-semibold text-gray-800 mb-2">Step 3: Matrix Multiplication</h5>
                      <div className="bg-green-50 rounded p-3 border border-green-200">
                        <div className="font-mono text-sm space-y-2">
                          <div className="text-center">
                            <div className="inline-flex items-center gap-2">
                              <div className="border-2 border-purple-300 rounded p-2">
                                {transformationMatrix.map((row, i) => (
                                  <div key={i} className="text-center">
                                    [{row.map((val, j) => (
                                      <span key={j} className={j < row.length - 1 ? 'mr-2' : ''}>
                                        {val.toFixed(3)}
                                      </span>
                                    ))}]
                                  </div>
                                ))}
                              </div>
                              <span className="text-xl">×</span>
                              <div className="border-2 border-blue-300 rounded p-2">
                                <div>[{originalHomogeneous[0].toFixed(2)}]</div>
                                <div>[{originalHomogeneous[1].toFixed(2)}]</div>
                                <div>[{originalHomogeneous[2]}]</div>
                              </div>
                              <span className="text-xl">=</span>
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 mt-3 space-y-1">
                            <div><strong>Row 1:</strong> {transformationMatrix[0][0].toFixed(3)} × {originalHomogeneous[0].toFixed(2)} + {transformationMatrix[0][1].toFixed(3)} × {originalHomogeneous[1].toFixed(2)} + {transformationMatrix[0][2].toFixed(3)} × {originalHomogeneous[2]} = {(transformationMatrix[0][0] * originalHomogeneous[0] + transformationMatrix[0][1] * originalHomogeneous[1] + transformationMatrix[0][2] * originalHomogeneous[2]).toFixed(3)}</div>
                            <div><strong>Row 2:</strong> {transformationMatrix[1][0].toFixed(3)} × {originalHomogeneous[0].toFixed(2)} + {transformationMatrix[1][1].toFixed(3)} × {originalHomogeneous[1].toFixed(2)} + {transformationMatrix[1][2].toFixed(3)} × {originalHomogeneous[2]} = {(transformationMatrix[1][0] * originalHomogeneous[0] + transformationMatrix[1][1] * originalHomogeneous[1] + transformationMatrix[1][2] * originalHomogeneous[2]).toFixed(3)}</div>
                            <div><strong>Row 3:</strong> {transformationMatrix[2][0].toFixed(3)} × {originalHomogeneous[0].toFixed(2)} + {transformationMatrix[2][1].toFixed(3)} × {originalHomogeneous[1].toFixed(2)} + {transformationMatrix[2][2].toFixed(3)} × {originalHomogeneous[2]} = {(transformationMatrix[2][0] * originalHomogeneous[0] + transformationMatrix[2][1] * originalHomogeneous[1] + transformationMatrix[2][2] * originalHomogeneous[2]).toFixed(3)}</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Step 4: Result */}
                    <div>
                      <h5 className="font-semibold text-gray-800 mb-2">Step 4: Transformed Point</h5>
                      <div className="bg-yellow-50 rounded p-3 border border-yellow-200">
                        <div className="font-mono text-sm">
                          <div className="text-center mb-2">
                            <div className="inline-block border-2 border-yellow-300 rounded p-2">
                              <div>[{transformedPoint[0].toFixed(3)}]</div>
                              <div>[{transformedPoint[1].toFixed(3)}]</div>
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 text-center">
                            Transformed Point: ({transformedPoint[0].toFixed(3)}, {transformedPoint[1].toFixed(3)})
                          </div>
                          <div className="text-xs text-gray-500 text-center mt-1">
                            (Note: The third coordinate is always 1 in homogeneous coordinates, so we drop it for the final 2D point)
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Summary */}
                    <div className="bg-indigo-50 rounded p-3 border border-indigo-200">
                      <div className="text-sm text-indigo-900">
                        <strong>Summary:</strong> Original point ({originalPoint[0].toFixed(2)}, {originalPoint[1].toFixed(2)}) 
                        → Transformed point ({transformedPoint[0].toFixed(3)}, {transformedPoint[1].toFixed(3)})
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

