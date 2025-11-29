<div align="center">

# Retro Neural Fusion
*Single-digit arithmetic networks colliding in a retro CRT sandbox.*

</div>

## Overview

Retro Neural Fusion is an interactive TensorFlow.js playground that trains two lightweight neural networks in the browser:

- **Network A (Addition)** learns one-digit addition (0‑9 + 0‑9).
- **Network B (Multiplication)** learns one-digit multiplication (0‑9 × 0‑9).

After every epoch the weights of A & B are averaged to spawn **Network C**, which is evaluated on one-digit division (1‑9 ÷ 1‑9). The UI presents:

- Live training stats (loss, accuracy, epoch counts) for each network.
- Retro-futuristic neuron visualizations showing connectivity strength and activation pulses.
- Micro bar charts for recent loss/accuracy trends.

Everything runs client-side with React, Vite, and TensorFlow.js—no backend required.

## Tech Stack

| Layer | Stack |
| --- | --- |
| UI | React 19 + TypeScript + Vite |
| Styling | Custom CSS (retro CRT aesthetic) |
| ML | TensorFlow.js (Sequential API, Adam optimizer) |

## Getting Started

```bash
# install deps
npm install

# start dev server
npm run dev

# lint (eslint + prettier)
npm run lint
```

Visit the printed URL (default `http://localhost:5173`) and hit **START TRAINING** to watch the experiment unfold.

## Training Pipeline

1. **Dataset generation**
   - Addition & multiplication datasets enumerate all ordered pairs (0‑9, 0‑9) → 100 samples each.
   - Division test set enumerates (1‑9, 1‑9) to avoid divide-by-zero.

2. **Model architecture**
   - `Dense(16, relu)` hidden layer + `Dense(1, linear)` output.
   - Adam optimizer, MSE loss, batch size 32, 50 epochs.

3. **Per-epoch routine**
   - Fit addition network for one epoch, compute training loss + tolerance-based accuracy (|error| ≤ 0.5 counts as correct).
   - Repeat for multiplication network.
   - Average the weight tensors (`(W_add + W_mul) / 2`) to create Network C.
   - Evaluate Network C on the division dataset, log loss/accuracy.

4. **Visualization**
   - Canvas renders neurons + strongest 30% of connections, color/size modulated by simulated activations during training.
   - Layer labels and sparkline-style bar charts render beneath the visual to keep everything in one viewport.

## Project Structure

```
src/
├── App.tsx               # Main training loop & UI panels
├── NetworkVisualization.tsx
├── App.css / index.css   # Retro styling + layout
└── main.tsx              # React entry point
```

- `NetworkVisualization.tsx` owns the canvas rendering logic and activity pulses.
- `App.tsx` houses dataset creation, training loop, stats tracking, and layout.

## Notable Implementation Details

- **Tensor hygiene:** remember to dispose tensors/models if you extend the training loop. (`averageWeights`, merged models, and dataset tensors are good places to double-check.)
- **Accuracy metric:** intentionally lenient (absolute error ≤ 0.5) because outputs are continuous scalars.
- **UI footprint:** tuned to fit on a single 1440×900 viewport without scrolling.

## Extending the Experiment

Ideas for future iterations:

- Swap averaging for knowledge distillation or meta-learning strategies.
- Introduce noise/regularization or curriculum schedules.
- Replace deterministic datasets with streamed samples from a worker.
- Hook up WebGPU (tfjs-backend-webgpu) for heavier models.

## License

MIT — build weird neural mashups and have fun.
