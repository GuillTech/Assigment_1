# DRL for Automated Testing: Tetris & Wordle

**Assignment 1 - Deep Reinforcement Learning for Automated Game Testing**

---

## Presentation
https://www.youtube.com/watch?v=jj8x_JYx__c

##  Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip 

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tetris
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

Core dependencies (see `requirements.txt` for full list):
- `gymnasium>=1.21.1` - RL environment interface
- `stable-baselines3>=2.0.0` - DRL algorithms
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data analysis
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `pygame>=2.5.0` - Tetris rendering
- `pyyaml>=6.0` - Configuration files

---

##  Reproduce models

### Train a Tetris Agent

```bash
# Train PPO agent with balanced persona
python src/tetris_train.py --algo ppo --play-style balanced --timesteps 100000 --seed 42

# Train with aggressive persona
python src/tetris_train.py --algo ppo --play-style aggressive --timesteps 100000 --seed 42

# Train with DQN
python src/tetris_train.py --algo dqn --play-style conservative --timesteps 100000 --seed 7
```

### Train a Wordle Agent

```bash
# Train PPO agent with explorer persona
python src/Wordle_train.py --algo ppo --persona explorer --timesteps 100000 --seed 42

# Train with speedrunner persona
python src/Wordle_train.py --algo ppo --persona speedrunner --timesteps 100000 --seed 42

# Train A2C with survivor persona
python src/Wordle_train.py --algo a2c --persona survivor --timesteps 100000 --seed 7

python src/Wordle_train.py --algo a2c --persona validator --timesteps 100000 --seed 7
```

### Evaluate Models

```bash
# Evaluate Tetris agent
python src/tetris_eval.py --model models/ppo_balanced_seed42.zip --episodes 100

# Evaluate Wordle agent
python src/wordle_eval.py --model models/ppo_explorer_seed42.zip --episodes 300

# Compare multiple models
python src/wordle_eval.py --model "models/a2c_*_seed7.zip" --episodes 500 --compare
```

### Visualize Agent Gameplay

```bash
# Visualize Tetris agent
python src/tetris_visualize.py --model_path models/ppo_balanced_seed42.zip --algo ppo --fps 60

# Visualize Wordle agent (GUI)
python src/wordle_visualize.py --model models/ppo_explorer_seed42.zip --episodes 10 --delay 1.0
```

---

##  Environments

### Tetris Environment

A  block-stacking game where you must place falling tetrominoes to clear lines and survive as long as possible.

**Action Space**: 6 discrete actions
- `0`: Move left
- `1`: Move right  
- `2`: Rotate clockwise
- `3`: Soft drop (move down)
- `4`: Hard drop (place immediately)
- `5`: No-op (do nothing)

**Observation Space**: 
- **Grid state** (200 floats): Flattened 20×10 board
- **Current piece** (7 floats): One-hot encoding (I, O, T, L, J, S, Z)
- **Next piece** (7 floats): One-hot encoding
- **Position** (2 floats): Normalized x, y coordinates
- **Aggregate features** (10 floats): Holes, heights, bumpiness, wells, etc.

**Total**: 226-dimensional continuous observation space

**Reward Personas**:
1. **Balanced**: Moderate penalties/rewards across all actions
2. **Aggressive**: High rewards for Tetrises (4-line clears), tolerates risk
3. **Conservative**: Heavy penalties for holes and height, prioritizes survival
4. **Speedrun**: Emphasizes efficiency and speed, bonus for combos

**Key Metrics**:
- Score, lines cleared (singles/doubles/triples/tetrises)
- Survival time, pieces placed
- Holes created, max height, bumpiness, wells
- Tetris rate, combo streaks
- Invalid action rate
- Board coverage, unique states visited

### Wordle Environment

A word-guessing puzzle where agents have 6 attempts to guess a 5-letter target word using feedback.

**Action Space**: Discrete (one action per valid word in dictionary, ~500 words)

**Observation Space**:
- **Feedback grid** (6×5): Previous guesses with color-coded feedback
- **Letter knowledge** (26): Per-letter status (unknown/absent/present/correct)
- **Guess count** (1): Number of guesses used

**Feedback Encoding**:
- `0`: UNKNOWN (not guessed yet)
- `1`: ABSENT (gray - letter not in word)
- `2`: PRESENT (yellow - letter in word, wrong position)
- `3`: CORRECT (green - letter in correct position)

**Reward Personas**:
1. **Explorer**: Prioritizes discovering new letters and positions
2. **Speedrunner**: Emphasizes winning quickly, high efficiency bonus
3. **Validator**: Balanced approach, avoids repeated/invalid guesses
4. **Survivor**: Conservative, focuses on gathering information safely

**Key Metrics**:
- Win rate, average guesses (overall and when won)
- Guess distribution (1-6 guesses)
- Repeated guesses, unique guesses
- Letters/positions discovered
- Hardest/easiest words
- Common winning sequences

---

##  Training

### Command-Line Arguments

Both training scripts support the following arguments:

```bash
# Algorithm selection
--algo {ppo,a2c,dqn}              # RL algorithm

# Environment configuration  
--play-style {balanced,aggressive,conservative,speedrun}  # Tetris only
--persona {explorer,speedrunner,validator,survivor}       # Wordle only

# Training parameters
--timesteps 100000                # Total training timesteps
--seed 42                         # Random seed for reproducibility
--n-envs 4                        # Number of parallel environments

# I/O parameters
--output-dir models               # Directory for saved models
--logs-dir logs                   # Directory for logs
--experiment-name custom_name     # Custom experiment identifier
--config path/to/config.yaml      # YAML config file

# Frequency parameters
--save-freq 50000                 # Model checkpoint frequency
--eval-freq 10000                 # Evaluation frequency
```
### Training Outputs

Each training run generates:
- **Model checkpoint**: `models/{algo}_{persona}_seed{seed}.zip`
- **Best model**: `models/{experiment}_best/best_model.zip`
- **Logs**: `logs/{experiment}/`
  - `config.yaml`: Experiment configuration
  - `episode_metrics.csv`: Per-episode data
  - `aggregate_metrics.json`: Summary statistics
  - `training_metrics.png`: Visualization plots
  - TensorBoard logs

---

##  Evaluation & Testing

### Command-Line Arguments

```bash
# Model specification
--model path/to/model.zip         # Path to trained model (supports wildcards)

# Evaluation parameters
--episodes 500                    # Number of evaluation episodes
--seed 42                         # Random seed

# Output
--output-dir eval_results         # Output directory
--compare                         # Generate comparison plots (multi-model)

```

### Evaluation Outputs

Each evaluation generates:
- **Episode data**: `eval_results/{model}/episodes.csv`
- **Metrics summary**: `eval_results/{model}/metrics.json`
- **Visualization plots**: `eval_results/{model}/evaluation_results.png`
- **Issue report**: `eval_results/issues_report.txt` (Tetris only)
- **Comparison plots**: `eval_results/model_comparison.png` (when using `--compare`)

### Automated Issue Detection (Tetris)

The evaluation framework automatically detects potential issues:

1. **High Invalid Action Rate**: Agent attempts invalid moves frequently (>20%)
2. **Low Line Clearing Efficiency**: <0.3 lines cleared per piece
3. **Excessive Hole Creation**: Average >10 holes per game
4. **Poor Height Management**: Average max height >15 (danger zone)
5. **Low Tetris Rate**: <5% of line clears are 4-line clears
6. **Repetitive Action Patterns**: Same action sequence repeated excessively
7. **Low Survival Time**: Games end too quickly (<100 steps)

Issues are categorized by severity (Low/Medium/High/Critical) with recommendations.

---

## Visualization

### Tetris Visualization

Watch a model play Tetris:

```bash
python src/tetris_visualize.py \
  --model_path models/ppo_balanced_seed42.zip \
  --algo ppo \
  --fps 60 \
  --seed 42
```



### Wordle Visualization
model solving Wordle puzzles:

```bash
python src/wordle_visualize.py \
  --model models/ppo_explorer_seed42.zip \
  --episodes 10 \
  --delay 1.0 \
  --persona explorer \
  --seed 42

---

## Metrics & Analysis

### Tetris Metrics (30+ tracked)

**Performance Metrics**:
- Score (current, max, average, percentiles)
- Lines cleared (total, singles, doubles, triples, tetrises)
- Survival time (steps before game over)
- Pieces placed
- Lines per piece (efficiency)
- Score per piece
- Tetris rate (4-line clears / total clears)

**Board State Metrics**:
- Holes created (cumulative)
- Max height reached
- Average height
- Bumpiness (column height variation)
- Wells created
- Board coverage (% of cells visited)

**Action Metrics**:
- Total actions taken
- Action distribution (left, right, rotate, soft drop, hard drop, no-op)
- Invalid actions (attempted but failed)
- Invalid action rate

**Advanced Metrics**:
- Max combo streak
- Unique board states visited
- State revisits (repeated configurations)

### Wordle Metrics (15+ tracked)

**Performance Metrics**:
- Win rate (overall)
- Total wins/losses
- Average guesses (overall, when won, when lost)
- Guess distribution (1-6 guesses)
- Average reward per episode

**Efficiency Metrics**:
- Unique guesses per game
- Repeated guesses (wasted attempts)
- Letters discovered
- Positions confirmed

**Word-Level Analysis**:
- Hardest words (lowest win rate)
- Easiest words (highest win rate)
- Most common winning sequences

### Metrics Export Formats

**CSV** (`episodes.csv`):
```csv
episode,score,lines_cleared,survival_time,holes_created,max_height,...
0,1240,12,342,8,14,...
1,3580,35,891,12,16,...
```

**JSON** (`metrics.json`):
```json
{
  "total_episodes": 500,
  "avg_score": 2847.3,
  "max_score": 12450,
  "win_rate": 0.73,
  "guess_distribution": {
    "1": 5,
    "2": 43,
    "3": 189,
  }
}
```


---
## Model Results

## Tetris
-- ppo Aggressive Switches between left and right, trying to make two towers.
-- ppo Balanced mostly rotates in place and does nothing, 
-- a2c conservative Alternates left and right every action trying to spread the pieces as much as possible

## Worlde
-- ppo Explorer likes guessing the same thing twice in their 4th and 5th guess
-- validator and survivor guesses the same thing 5 times.
-- 
##  Reproducibility

All experiments are fully reproducible by design.

### Fixed Seeds

Every training and evaluation run uses explicit seeds:
```
--seed 42  # Controls: random.seed(), np.random.seed(), env.action_space.seed()
```

### Saved Configurations

Each experiment saves its full configuration:
```yaml
# logs/{experiment}/config.yaml
algo: ppo
play_style: balanced
timesteps: 100000
seed: 42
n_envs: 4
...
```

### Reproduction Steps

To reproduce any result:

1. **Install exact dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Use the exact training command** (from logs or paper):
   ```
   python src/tetris_train.py --algo ppo --play-style balanced --timesteps 100000 --seed 42 --n-envs 4
   ```

3. **Evaluate with the same seed**:
   ```
   python src/tetris_eval.py --model models/ppo_balanced_seed42.zip --episodes 500 --seed 100
   ```

4. **Check logs** for configuration verification:
   ```bash
   cat logs/ppo_balanced_seed42/config.yaml
   ```

---

##  Authors

**Group Members**: Guillermo Rebolledo 100865463
