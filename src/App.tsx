import { useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { NetworkVisualization } from './NetworkVisualization';
import './App.css';

interface NetworkStats {
  epoch: number;
  loss: number;
  accuracy: number;
  testLoss?: number;
  testAccuracy?: number;
}

function App() {
  const [additionNet, setAdditionNet] = useState<tf.Sequential | null>(null);
  const [multiplicationNet, setMultiplicationNet] = useState<tf.Sequential | null>(null);
  const [mergedNet, setMergedNet] = useState<tf.Sequential | null>(null);
  
  const [additionStats, setAdditionStats] = useState<NetworkStats[]>([]);
  const [multiplicationStats, setMultiplicationStats] = useState<NetworkStats[]>([]);
  const [mergedStats, setMergedStats] = useState<NetworkStats[]>([]);
  
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  // Create a simple neural network
  const createNetwork = useCallback(() => {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [2], units: 16, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'linear' })
      ]
    });
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError', metrics: ['mse'] });
    return model;
  }, []);

  // Generate training data for addition (0-9 + 0-9)
  const generateAdditionData = useCallback(() => {
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let a = 0; a <= 9; a++) {
      for (let b = 0; b <= 9; b++) {
        inputs.push([a, b]);
        outputs.push(a + b);
      }
    }
    return { inputs: tf.tensor2d(inputs), outputs: tf.tensor2d(outputs, [outputs.length, 1]) };
  }, []);

  // Generate training data for multiplication (0-9 * 0-9)
  const generateMultiplicationData = useCallback(() => {
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let a = 0; a <= 9; a++) {
      for (let b = 0; b <= 9; b++) {
        inputs.push([a, b]);
        outputs.push(a * b);
      }
    }
    return { inputs: tf.tensor2d(inputs), outputs: tf.tensor2d(outputs, [outputs.length, 1]) };
  }, []);

  // Generate test data for division (1-9 / 1-9, avoiding division by zero)
  const generateDivisionData = useCallback(() => {
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let a = 1; a <= 9; a++) {
      for (let b = 1; b <= 9; b++) {
        inputs.push([a, b]);
        outputs.push(a / b);
      }
    }
    return { inputs: tf.tensor2d(inputs), outputs: tf.tensor2d(outputs, [outputs.length, 1]) };
  }, []);

  // Average weights of two networks
  const averageWeights = useCallback((net1: tf.Sequential, net2: tf.Sequential): tf.Sequential => {
    const merged = createNetwork();
    const weights1 = net1.getWeights();
    const weights2 = net2.getWeights();
    const averagedWeights = weights1.map((w1, i) => {
      const w2 = weights2[i];
      return tf.tidy(() => {
        return w1.add(w2).div(2);
      });
    });
    merged.setWeights(averagedWeights);
    return merged;
  }, [createNetwork]);

  // Calculate accuracy (within 0.5 tolerance)
  const calculateAccuracy = useCallback((predictions: tf.Tensor, targets: tf.Tensor): number => {
    return tf.tidy(() => {
      const diff = predictions.sub(targets).abs();
      const correct = diff.lessEqual(0.5).sum().dataSync()[0];
      return correct / predictions.shape[0];
    });
  }, []);

  // Training function
  const trainNetworks = useCallback(async () => {
    if (isTraining) return;
    
    setIsTraining(true);
    setCurrentEpoch(0);
    setAdditionStats([]);
    setMultiplicationStats([]);
    setMergedStats([]);

    // Initialize networks
    const addNet = createNetwork();
    const multNet = createNetwork();
    setAdditionNet(addNet);
    setMultiplicationNet(multNet);

    const addData = generateAdditionData();
    const multData = generateMultiplicationData();
    const divData = generateDivisionData();

    const epochs = 50;
    const batchSize = 32;

    for (let epoch = 0; epoch < epochs; epoch++) {
      setCurrentEpoch(epoch + 1);

      // Train addition network
      const addHistory = await addNet.fit(addData.inputs, addData.outputs, {
        epochs: 1,
        batchSize,
        verbose: 0
      });
      const addLoss = addHistory.history.loss[0] as number;
      const addPred = addNet.predict(addData.inputs) as tf.Tensor;
      const addAcc = calculateAccuracy(addPred, addData.outputs);
      addPred.dispose();
      setAdditionStats(prev => [...prev, { epoch: epoch + 1, loss: addLoss, accuracy: addAcc }]);

      // Train multiplication network
      const multHistory = await multNet.fit(multData.inputs, multData.outputs, {
        epochs: 1,
        batchSize,
        verbose: 0
      });
      const multLoss = multHistory.history.loss[0] as number;
      const multPred = multNet.predict(multData.inputs) as tf.Tensor;
      const multAcc = calculateAccuracy(multPred, multData.outputs);
      multPred.dispose();
      setMultiplicationStats(prev => [...prev, { epoch: epoch + 1, loss: multLoss, accuracy: multAcc }]);

      // Create merged network and test on division
      const merged = averageWeights(addNet, multNet);
      
      // Dispose previous merged network
      if (mergedNet) {
        mergedNet.dispose();
      }
      setMergedNet(merged);
      
      const divPred = merged.predict(divData.inputs) as tf.Tensor;
      const divLoss = tf.tidy(() => {
        return tf.losses.meanSquaredError(divData.outputs, divPred);
      });
      const divLossVal = divLoss.dataSync()[0];
      divLoss.dispose();
      const divAcc = calculateAccuracy(divPred, divData.outputs);
      divPred.dispose();
      setMergedStats(prev => [...prev, { 
        epoch: epoch + 1, 
        loss: divLossVal, 
        accuracy: divAcc,
        testLoss: divLossVal,
        testAccuracy: divAcc
      }]);

      // Small delay for UI update
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Cleanup
    addData.inputs.dispose();
    addData.outputs.dispose();
    multData.inputs.dispose();
    multData.outputs.dispose();
    divData.inputs.dispose();
    divData.outputs.dispose();

    setIsTraining(false);
  }, [isTraining, createNetwork, generateAdditionData, generateMultiplicationData, generateDivisionData, averageWeights, calculateAccuracy]);

  // Initialize networks on mount
  useEffect(() => {
    const addNet = createNetwork();
    const multNet = createNetwork();
    setAdditionNet(addNet);
    setMultiplicationNet(multNet);
    
    return () => {
      addNet.dispose();
      multNet.dispose();
    };
  }, [createNetwork]);

  // Cleanup merged network on unmount
  useEffect(() => {
    return () => {
      if (mergedNet) {
        mergedNet.dispose();
      }
    };
  }, [mergedNet]);

  const getLatestStats = (stats: NetworkStats[]) => stats[stats.length - 1] || { epoch: 0, loss: 0, accuracy: 0 };

  return (
    <div className="app">
      <div className="header">
        <h1 className="title">NEURAL NETWORK FUSION</h1>
        <div className="subtitle">RETRO NEURAL EXPERIMENT</div>
      </div>

      <div className="controls">
        <button 
          className="control-btn" 
          onClick={trainNetworks} 
          disabled={isTraining}
        >
          {isTraining ? `TRAINING... EPOCH ${currentEpoch}/50` : 'START TRAINING'}
        </button>
      </div>

      <div className="networks-container">
        <div className="network-panel">
          <div className="panel-header">NETWORK A: ADDITION</div>
          <div className="panel-content">
            <NetworkVisualization 
              network={additionNet} 
              color="#00ff41" 
              width={280} 
              height={350}
              updateTrigger={currentEpoch}
              isTraining={isTraining}
            />
            <div className="stat-row">
              <span className="stat-label">EPOCH:</span>
              <span className="stat-value">{getLatestStats(additionStats).epoch}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">LOSS:</span>
              <span className="stat-value">{getLatestStats(additionStats).loss.toFixed(4)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">ACCURACY:</span>
              <span className="stat-value">{(getLatestStats(additionStats).accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="chart">
              <div className="chart-label">LOSS TREND</div>
              <div className="chart-bars">
                {additionStats.slice(-10).map((stat, i) => {
                  const maxLoss = Math.max(...additionStats.map(s => s.loss), 0.1);
                  const normalizedHeight = Math.max(10, (stat.loss / maxLoss) * 100);
                  return (
                    <div 
                      key={i} 
                      className="chart-bar" 
                      style={{ height: `${normalizedHeight}%` }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        <div className="network-panel">
          <div className="panel-header">NETWORK B: MULTIPLICATION</div>
          <div className="panel-content">
            <NetworkVisualization 
              network={multiplicationNet} 
              color="#00ff41" 
              width={280} 
              height={350}
              updateTrigger={currentEpoch}
              isTraining={isTraining}
            />
            <div className="stat-row">
              <span className="stat-label">EPOCH:</span>
              <span className="stat-value">{getLatestStats(multiplicationStats).epoch}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">LOSS:</span>
              <span className="stat-value">{getLatestStats(multiplicationStats).loss.toFixed(4)}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">ACCURACY:</span>
              <span className="stat-value">{(getLatestStats(multiplicationStats).accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="chart">
              <div className="chart-label">LOSS TREND</div>
              <div className="chart-bars">
                {multiplicationStats.slice(-10).map((stat, i) => {
                  const maxLoss = Math.max(...multiplicationStats.map(s => s.loss), 0.1);
                  const normalizedHeight = Math.max(10, (stat.loss / maxLoss) * 100);
                  return (
                    <div 
                      key={i} 
                      className="chart-bar" 
                      style={{ height: `${normalizedHeight}%` }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        <div className="network-panel merged">
          <div className="panel-header">NETWORK C: MERGED (TESTED ON DIVISION)</div>
          <div className="panel-content">
            <NetworkVisualization 
              network={mergedNet} 
              color="#ff6b00" 
              width={280} 
              height={350}
              updateTrigger={currentEpoch}
              isTraining={isTraining}
            />
            <div className="stat-row">
              <span className="stat-label">EPOCH:</span>
              <span className="stat-value">{getLatestStats(mergedStats).epoch}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">TEST LOSS:</span>
              <span className="stat-value">{getLatestStats(mergedStats).testLoss?.toFixed(4) || '0.0000'}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">TEST ACCURACY:</span>
              <span className="stat-value">{(getLatestStats(mergedStats).testAccuracy ? getLatestStats(mergedStats).testAccuracy! * 100 : 0).toFixed(1)}%</span>
            </div>
            <div className="chart">
              <div className="chart-label">ACCURACY TREND</div>
              <div className="chart-bars">
                {mergedStats.slice(-10).map((stat, i) => {
                  const accuracy = stat.testAccuracy || 0;
                  const normalizedHeight = Math.max(10, accuracy * 100);
                  return (
                    <div 
                      key={i} 
                      className="chart-bar accuracy-bar" 
                      style={{ height: `${normalizedHeight}%` }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
