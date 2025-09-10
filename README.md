# Artificial Intelligence Course Assignment

This repository contains my comprehensive AI course assignment covering fundamental machine learning concepts, reinforcement learning algorithms, and advanced AI techniques. The project is structured into three main parts, each focusing on different aspects of artificial intelligence.

## Project Overview

This assignment demonstrates practical implementation and analysis of various AI algorithms across three distinct domains:

- **Part A**: Supervised Learning and Unsupervised Learning
- **Part B**: Reinforcement Learning (Policy and Value Iteration, Monte Carlo, Temporal Difference)
- **Part C**: Advanced AI Techniques (Computer Vision, Dimensionality Reduction, Q-Learning)

## Project Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── SEMTM0016 PartA/         # Supervised & Unsupervised Learning
│   ├── Q1.ipynb            # Classification: Decision Trees, KNN, Logistic Regression
│   ├── Q2.ipynb            # Regression: Linear Regression Analysis
│   ├── Q3.ipynb            # Clustering: K-Means, GMM, PCA
│   ├── sprites_greyscale10.npz           # Classification dataset
│   ├── dungeon_sensorstats_train.csv    # Regression training data
│   ├── dungeon_sensorstats_test.csv     # Regression test data
│   └── dungeon_sensorstats.csv          # Clustering dataset
├── SEMTM0016 PartB/         # Reinforcement Learning
│   ├── Q1.py               # Policy Evaluation and Basic RL
│   ├── Q2.py               # Policy Iteration vs Value Iteration
│   ├── Q3.py               # Monte Carlo vs Temporal Difference Learning
│   ├── Q3_withMCUpdate.py  # Monte Carlo with updates
│   ├── core/               # Environment and grid utilities
│   │   ├── dungeonworld_grid.py
│   │   └── dungeonworld_objects.py
│   └── envs/               # Dungeon maze environment
│       └── simple_dungeonworld_env.py
└── SEMTM0016 PartC/         # Advanced AI Techniques
│   ├── Q1.py               # Computer Vision: KNN Classification on Images
│   ├── Q2.py               # Dimensionality Reduction: PCA + Clustering
│   ├── Q3.py               # Q-Learning vs SARSA Comparison
│   ├── Q3_Additional_work.py # Enhanced Q-Learning with Reward Shaping
│   ├── dungeon_images_colour80/         # Image dataset for computer vision
│   ├── dungeon_sensorstats_partC.csv   # Sensor data for Part C
│   ├── core/               # Environment utilities
│   │   ├── dungeonworld_grid.py
│   │   └── dungeonworld_objects.py
│   └── envs/               # Dungeon maze environment
│       └── simple_dungeonworld_env.py
```

## Part A: Supervised & Unsupervised Learning

### Q1: Classification Analysis
- **Dataset**: Sprite character classification (halfling vs human)
- **Algorithms Implemented**:
  - Decision Tree Classifier with hyperparameter tuning
  - K-Nearest Neighbors (KNN) with cross-validation
  - Logistic Regression with L2 regularization
- **Key Results**:
  - Best Decision Tree: 165 leaf nodes (95.73% accuracy)
  - Best KNN: k=1 (96.19% validation accuracy, 94.52% test accuracy)
  - Best Logistic Regression: C=1000 (89.39% test accuracy)

### Q2: Regression Analysis
- **Dataset**: Dungeon sensor statistics (human race subset)
- **Features**: Intelligence, stench, sound, heat → predicting bribe amount
- **Analysis**:
  - Multi-feature linear regression (MSE: 4.37)
  - Single-feature regression on intelligence (MSE: 13.77)
  - Demonstrates importance of feature selection

### Q3: Clustering Analysis
- **Dataset**: Height and weight measurements
- **Algorithms**:
  - K-Means clustering with elbow method and silhouette analysis
  - Gaussian Mixture Models (GMM) with AIC/BIC selection
  - Voronoi diagram visualization
- **Results**: Optimal k=4 clusters identified through multiple criteria

## Part B: Reinforcement Learning

### Q1: Policy Evaluation
- **Environment**: 8x8 Dungeon Maze
- **Policies Tested**:
  - Random policy
  - All-forward policy
  - Custom camera-based policy
- **Implementation**: Trajectory rollouts with reward analysis

### Q2: Policy vs Value Iteration
- **Algorithms**: 
  - Policy Iteration (policy evaluation + improvement)
  - Value Iteration (direct value function updates)
- **Analysis**:
  - Convergence comparison across different initializations
  - Performance metrics: iterations, computation time
  - MSE convergence tracking
- **Key Findings**: Value iteration generally faster, policy iteration more stable

### Q3: Monte Carlo vs Temporal Difference
- **Environment**: 10x10 Dungeon Maze
- **Algorithms**:
  - Monte Carlo (episode-based learning)
  - Temporal Difference (step-by-step learning)
- **Analysis**:
  - Learning curves and convergence
  - Value function visualization
  - MSE convergence comparison
- **Results**: TD learning shows faster convergence, MC more stable

## Part C: Advanced AI Techniques

### Q1: Computer Vision Classification
- **Dataset**: 80x80 color dungeon character images
- **Classes**: halfling, human, lizard, orc, wingedrat
- **Algorithm**: K-Nearest Neighbors with cross-validation
- **Features**: Flattened pixel values (19,200 features)
- **Results**: Optimal k=1, comprehensive confusion matrix analysis

### Q2: Dimensionality Reduction & Clustering
- **Dataset**: 9D dungeon sensor statistics
- **Pipeline**:
  - PCA reduction to 2D
  - K-Means clustering (elbow + silhouette analysis)
  - Gaussian Mixture Models (AIC/BIC selection)
- **Visualization**: 2D scatter plots with cluster assignments

### Q3: Q-Learning vs SARSA
- **Environment**: 10x10 Dungeon Maze
- **Algorithms**:
  - Q-Learning (off-policy)
  - SARSA (on-policy)
- **Analysis**:
  - Learning curves and convergence speed
  - Q-value distribution analysis
  - Reward per step efficiency
- **Results**: Q-Learning shows faster convergence, SARSA more conservative

### Q3 Additional Work: Enhanced Q-Learning
- **Improvements**:
  - Reward shaping based on distance to goal
  - Larger environment (14x14 grid)
  - Extended training (2000 episodes)
- **Results**: Significant improvement in learning efficiency and goal-reaching

## Technical Implementation

### Environment Details
- **Grid-based maze navigation**
- **Robot with position and direction state**
- **Camera view for partial observability**
- **Reward structure**: -1 per step, +100 for goal, -100 for walls**

## Key Results Summary

| Part | Algorithm | Best Performance | Key Insight |
|------|-----------|------------------|-------------|
| A-Q1 | KNN (k=1) | 96.19% accuracy | Simple algorithms can be highly effective |
| A-Q2 | Multi-feature LR | MSE: 4.37 | Feature selection crucial for performance |
| A-Q3 | K-Means (k=4) | Optimal clustering | Multiple criteria needed for cluster selection |
| B-Q2 | Value Iteration | Faster convergence | More efficient than policy iteration |
| B-Q3 | Temporal Difference | Faster learning | Step-by-step updates more efficient |
| C-Q1 | KNN (k=1) | High accuracy | Image classification with simple features |
| C-Q3 | Q-Learning | Better exploration | Off-policy learning more efficient |

## Learning Outcomes

This project demonstrates mastery of:

1. **Supervised Learning**: Classification and regression with hyperparameter tuning
2. **Unsupervised Learning**: Clustering algorithms and dimensionality reduction
3. **Reinforcement Learning**: Policy evaluation, value iteration, and temporal difference learning
4. **Computer Vision**: Image classification and feature extraction
5. **Algorithm Comparison**: Systematic evaluation and analysis of different approaches
6. **Visualization**: Comprehensive plotting and analysis of results

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Code
1. **Part A**: Open Jupyter notebooks in `SEMTM0016 PartA/`
   - All data files are included in the same directory
   - Run cells sequentially in each notebook
2. **Part B**: Run Python scripts in `SEMTM0016 PartB/`
   - Environment files (`core/` and `envs/`) are included
   - Execute: `python Q1.py`, `python Q2.py`, `python Q3.py`
3. **Part C**: Execute Python files in `SEMTM0016 PartC/`
   - Image dataset and sensor data are included
   - Execute: `python Q1.py`, `python Q2.py`, `python Q3.py`

### Environment Setup
All required files and datasets are included in their respective directories. No additional setup required beyond installing dependencies.

## Performance Analysis

Each part includes comprehensive analysis of:
- **Convergence behavior** and learning curves
- **Hyperparameter sensitivity** and optimization
- **Algorithm comparison** with statistical significance
- **Visualization** of results and decision boundaries
- **Computational efficiency** and scalability

## Future Enhancements

Potential improvements for each part:
- **Part A**: Deep learning approaches, feature engineering
- **Part B**: Deep Q-Networks, policy gradient methods
- **Part C**: Convolutional neural networks, advanced RL algorithms

---