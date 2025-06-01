## ICRL (In-Context Reinforcement Learning) Fine-tuning

In-Context Reinforcement Learning with PPO and Value Head

Example Training Steps with RWKV-7 G1 Model

### Overview

ICRL extends traditional RL to leverage in-context learning by structuring episodes as "k exemplars + n query-answer pairs" within a single context window. This approach enables the model to learn from both historical demonstrations and current feedback within the same episode.

### Key Features

- **Value Head**: Adds a single linear layer on top of final hidden state for variance-reduced policy gradients
- **Episodic Structure**: Episodes consist of k shaved-down exemplars + n query-answer pairs
- **PPO with KL Constraint**: Maintains closeness to base model while improving long-horizon rewards
- **LoRA/Adapter Protection**: Updates only LoRA adapters or last N blocks to preserve language knowledge
- **Exploration Enhancements**: Entropy bonus on hidden-state norm, episodic randomization, ε-mixture sampling

### Step.-1 Download Model
  - make directory, myfolder/models
  - download RWKV-x070-7B-20240816-ctx4096.pth from HF
  - copy to myfolder/models/RWKV-x070-7B-20240816-ctx4096.pth

### Step.0 Prepare Dataset

#### Data Format Requirements

**ICRL JSON-L Format:**
Each line should contain:
```json
{
  "exemplars": [
    {
      "prompt": "Question: What is 2+2?",
      "answer": "Answer: 4"
    },
    {
      "prompt": "Question: What is the capital of France?", 
      "answer": "Answer: Paris"
    }
  ],
  "queries": [
    {
      "prompt": "Question: What is 5*6?",
      "preferred_answer": "Answer: 30",
      "rejected_answer": "Answer: 25"
    }
  ],
  "episode_id": "unique_episode_identifier",
  "task_type": "math" // optional task categorization
}
```

**Key Requirements:**
- **k exemplars**: High-quality demonstration pairs (typically 2-5 pairs)
- **n queries**: Query-answer pairs for learning (typically 1-3 pairs)
- **Shaved-down exemplars**: Exemplars should be concise to save context space
- **Episode structure**: All data within one episode fits in context window

#### Generate JSONL:
```bash
./example/ICRL/step-0-prepare-jsonl.sh
```

### Step.1 Generate Tokenized File
```bash
./example/ICRL/step-1-jsonltoh5.sh
```

### Step.2 Supervised Bootstrap (Stage 1)
Short training run on mixed demonstration→query corpus with KL penalty:
```bash
./example/ICRL/step-2-bootstrap.sh
```

### Step.3 Reward Model Training (Stage 2)
Train separate reward model on pairwise comparisons:
```bash
./example/ICRL/step-3-reward-model.sh
```

### Step.4 PPO Fine-tuning (Stage 3-4)
Main ICRL training with PPO and exploration fixes:
```bash
./example/ICRL/step-4-icrl-ppo.sh
```

### Step.5 Test-time Adaptive Optimization (Optional Stage 5)
Per-request adaptation with synthetic rollouts:
```bash
./example/ICRL/step-5-test-time-adapt.sh
```

### Training Parameters

**Important ICRL Parameters:**
- `--icrl`: Enable ICRL mode (default: 0)
- `--icrl_k_exemplars`: Number of exemplars per episode (default: 3)
- `--icrl_n_queries`: Number of query pairs per episode (default: 1)
- `--icrl_episode_length`: Maximum tokens per episode (default: 2048)
- `--icrl_value_head`: Enable value head (default: 1)
- `--icrl_ppo_epochs`: PPO epochs per batch (default: 4)
- `--icrl_kl_beta`: KL penalty coefficient (default: 0.01)
- `--icrl_entropy_beta`: Entropy bonus coefficient (default: 0.01)
- `--icrl_epsilon_mixture`: ε-mixture probability (default: 0.1)
- `--icrl_reward_model_path`: Path to trained reward model

### Expected Performance

ICRL should provide:
- Better sample efficiency than standard RL
- Improved in-context learning capabilities
- Reduced catastrophic forgetting
- Enhanced exploration in complex reasoning tasks

### Notes

- Requires sufficient VRAM for value head and policy/reference model
- Works best with programmatic tasks or tasks with clear reward signals
- Episodic randomization helps prevent overfitting to task order
- Test-time adaptation is optional but can improve per-user performance

For questions or issues, please refer to the main repository documentation or open an issue. 