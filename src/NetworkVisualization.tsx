import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

interface NetworkVisualizationProps {
  network: tf.Sequential | null;
  color: string;
  width: number;
  height: number;
  updateTrigger?: number; // Force re-render when this changes
  isTraining?: boolean; // Show activity during training
}

export function NetworkVisualization({ network, color, width, height, updateTrigger, isTraining }: NetworkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activations, setActivations] = useState<number[][]>([]);
  const animationFrameRef = useRef<number>();

  // Simulate activations for activity visualization
  useEffect(() => {
    if (!network || !isTraining) {
      setActivations([]);
      return;
    }

    // Generate simulated activations that pulse through the network
    const generateActivations = () => {
      const layers = [2, 16, 16, 1];
      const time = Date.now() / 1000;
      return layers.map((size, layerIdx) => 
        Array.from({ length: size }, (_, nodeIdx) => {
          // Create a wave pattern that flows through layers
          const phase = (time * 0.5 + layerIdx * 0.3 + nodeIdx * 0.1) % (Math.PI * 2);
          return 0.3 + 0.4 * Math.sin(phase);
        })
      );
    };

    const interval = setInterval(() => {
      setActivations(generateActivations());
    }, 100);

    setActivations(generateActivations());

    return () => clearInterval(interval);
  }, [network, isTraining, updateTrigger]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, 0, width, height);

      if (!network) {
        // Draw placeholder when network is not available
        ctx.fillStyle = color;
        ctx.font = '10px Orbitron';
        ctx.textAlign = 'center';
        ctx.fillText('NETWORK NOT INITIALIZED', width / 2, height / 2);
        return;
      }

      try {
        const weights = network.getWeights();
        
        // Network architecture: [2] -> [16] -> [1]
        const layers = [2, 16, 1];
        const layerSpacing = width / (layers.length + 1);
        const nodeRadius = 4;
        
        // Calculate node positions
        const nodePositions: { x: number; y: number }[][] = [];
        
        layers.forEach((nodeCount, layerIdx) => {
          const x = layerSpacing * (layerIdx + 1);
          const positions: { x: number; y: number }[] = [];
          const verticalSpacing = height / (nodeCount + 1);
          
          for (let i = 0; i < nodeCount; i++) {
            positions.push({
              x,
              y: verticalSpacing * (i + 1)
            });
          }
          nodePositions.push(positions);
        });

        // Draw connections
        // Weights come in pairs: [kernel, bias] for each layer
        let weightPairIdx = 0;
        for (let layerIdx = 0; layerIdx < layers.length - 1; layerIdx++) {
          const fromLayer = nodePositions[layerIdx];
          const toLayer = nodePositions[layerIdx + 1];
          
          // Get kernel weights (first of each pair)
          const kernelIdx = weightPairIdx * 2;
          if (kernelIdx < weights.length) {
            const weightMatrix = weights[kernelIdx];
            const weightData = weightMatrix.dataSync();
            const weightShape = weightMatrix.shape;
            
            // Weight matrix shape: [fromNodes, toNodes] - data is row-major
            // So data[fromIdx * toNodes + toIdx] gives weight from fromIdx to toIdx
            const fromNodes = weightShape[0];
            const toNodes = weightShape[1];
            
            // Collect all weights for this layer to find top connections
            const connectionWeights: Array<{fromIdx: number, toIdx: number, weight: number, absWeight: number}> = [];
            for (let fromIdx = 0; fromIdx < Math.min(fromLayer.length, fromNodes); fromIdx++) {
              for (let toIdx = 0; toIdx < Math.min(toLayer.length, toNodes); toIdx++) {
                const dataIdx = fromIdx * toNodes + toIdx;
                const weight = weightData[dataIdx];
                const absWeight = Math.abs(weight);
                connectionWeights.push({ fromIdx, toIdx, weight, absWeight });
              }
            }
            
            // Sort by absolute weight and only show top 30% of connections
            connectionWeights.sort((a, b) => b.absWeight - a.absWeight);
            const topConnections = Math.max(
              Math.ceil(connectionWeights.length * 0.3),
              Math.min(fromLayer.length, toLayer.length) // At least show one connection per node
            );
            const threshold = connectionWeights[topConnections - 1]?.absWeight || 0;
            
            // Parse hex color to RGB
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            
            // Draw only top connections
            for (let i = 0; i < topConnections; i++) {
              const conn = connectionWeights[i];
              if (conn.absWeight < threshold * 0.5) break; // Stop if weight drops significantly
              
              const normalizedWeight = Math.min(conn.absWeight / 2, 1);
              
              ctx.strokeStyle = conn.weight >= 0 
                ? `rgba(${r}, ${g}, ${b}, ${0.15 + normalizedWeight * 0.25})`
                : `rgba(255, 50, 50, ${0.15 + normalizedWeight * 0.25})`;
              ctx.lineWidth = normalizedWeight * 0.8 + 0.2;
              
              ctx.beginPath();
              ctx.moveTo(fromLayer[conn.fromIdx].x, fromLayer[conn.fromIdx].y);
              ctx.lineTo(toLayer[conn.toIdx].x, toLayer[conn.toIdx].y);
              ctx.stroke();
            }
            weightPairIdx++;
          }
        }

        // Draw nodes (neurons) with activity indicators
        const time = Date.now() / 1000;
        nodePositions.forEach((layer, layerIdx) => {
          layer.forEach((pos, nodeIdx) => {
            // Get activation value for this neuron
            let activation = 0.5; // Default
            if (activations.length > layerIdx && activations[layerIdx].length > nodeIdx) {
              activation = activations[layerIdx][nodeIdx];
            } else if (isTraining) {
              // Animate with sine wave if no activations available
              activation = 0.5 + 0.3 * Math.sin(time * 2 + layerIdx + nodeIdx);
            }
            
            // Normalize activation to 0-1
            activation = Math.max(0, Math.min(1, activation));
            
            // Calculate glow intensity based on activation
            const glowIntensity = isTraining ? activation : 0.5;
            const nodeSize = nodeRadius * (1 + glowIntensity * 0.5);
            
            // Outer glow with pulsing effect
            const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, nodeSize * 3);
            const r = parseInt(color.slice(1, 3), 16);
            const g = parseInt(color.slice(3, 5), 16);
            const b = parseInt(color.slice(5, 7), 16);
            
            gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${0.8 * glowIntensity})`);
            gradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${0.4 * glowIntensity})`);
            gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, nodeSize * 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Inner node with activation-based brightness
            const brightness = 0.5 + glowIntensity * 0.5;
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${brightness})`;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, nodeSize, 0, Math.PI * 2);
            ctx.fill();
            
            // Glow effect
            ctx.shadowBlur = 8 * glowIntensity;
            ctx.shadowColor = color;
            ctx.fill();
            ctx.shadowBlur = 0;
          });
        });

        // Don't draw labels on canvas - they'll be HTML below
      } catch (error) {
        console.error('Error visualizing network:', error);
        ctx.fillStyle = color;
        ctx.font = '10px Orbitron';
        ctx.textAlign = 'center';
        ctx.fillText('VISUALIZATION ERROR', width / 2, height / 2);
      }
    };

    draw();
    
    // Animate during training
    if (isTraining) {
      const animate = () => {
        draw();
        animationFrameRef.current = requestAnimationFrame(animate);
      };
      animationFrameRef.current = requestAnimationFrame(animate);
      
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }
  }, [network, color, width, height, updateTrigger, isTraining, activations]);

  const layerNames = ['INPUT', 'PROCESSING', 'OUTPUT'];
  const layerDescriptions = ['(2 nums)', 'LAYER', '(result)'];

  return (
    <div className="network-viz-container">
      <div className="viz-label">NEURAL ARCHITECTURE</div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="network-canvas"
      />
      <div className="layer-labels">
        {layerNames.map((name, idx) => (
          <div key={idx} className="layer-label">
            <div className="layer-label-main">{name}</div>
            <div className="layer-label-desc">{layerDescriptions[idx]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

