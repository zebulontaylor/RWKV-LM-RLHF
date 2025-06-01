import os
import copy
import re
import torch
from google import genai
from .trainutils import L2Wrap


def add_icrl_value_head(model, args):
    # No additional value head needed for now
    return


def setup_icrl_models(model, args):
    # Create a frozen reference copy for PPO baseline
    model.ref_model = copy.deepcopy(model)
    model.ref_model.eval()
    # Setup Gemini client for reward scoring
    api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    model.icrl_client = genai.Client(api_key=api_key)
    model.icrl_gemini_model = args.gemini_model


def patch_icrl_methods(cls):
    # Bootstrap phase (optional) just defer to PPO
    def training_step_icrl_bootstrap(self, batch, batch_idx):
        return self.training_step_icrl_ppo(batch, batch_idx)

    # Reward model training (optional) just defer to PPO
    def training_step_icrl_reward_model(self, batch, batch_idx):
        return self.training_step_icrl_ppo(batch, batch_idx)

    # Main PPO training step for ICRL
    def training_step_icrl_ppo(self, batch, batch_idx):
        """
        In-Context RL using PPO:
        - Generate a response from the model
        - Score it using Gemini API on the evaluation criterion
        - Compute a simple loss to encourage higher scores
        """
        # Multi-step ICL PPO over multiple questions per episode
        device = self.emb.weight.device
        assert len(batch) == 1, "ICRL micro-batch size must be 1"
        sample = batch[0]
        context = sample.get('context', '')
        questions = sample.get('questions', [])
        criterion = sample.get('evaluation_criterion', '')
        num_steps = min(len(questions), self.args.icrl_n_queries)
        prompt_so_far = context
        total_loss = torch.tensor(0.0, device=device)
        for i in range(num_steps):
            q = questions[i]
            # Build prompt with accumulated Q/A pairs
            prompt = f"{prompt_so_far}\n{q}\nA:"
            input_ids = torch.tensor(self.tokenizer.encode(prompt), device=device).unsqueeze(0)
            # Greedy generation
            logits, _ = self(input_ids)
            gen_ids = logits.argmax(dim=-1)[0].tolist()
            response_text = self.tokenizer.decode(gen_ids)
            # Judge prompt for this step
            judge_prompt = (
                f"Rate this response on a scale from 1 to 10 according to the following criterion: {criterion}"
                f"\nContext: {prompt_so_far}\nQuestion: {q}\nResponse: {response_text}\nJust return the numeric score."
            )
            # Query Gemini for reward
            score_text = self.icrl_client.models.generate_content(
                model=self.icrl_gemini_model,
                contents=judge_prompt
            ).text.strip()
            # Parse numeric score
            m = re.search(r"\d+(?:\.\d+)?", score_text)
            score = float(m.group()) if m else 0.0
            print(f"[ICRL PPO] Step {i+1}/{num_steps} Reward: {score}")
            # Accumulate negative reward as loss
            total_loss = total_loss + (-score)
            # Append Q/A to context for next step
            prompt_so_far = f"{prompt_so_far}\n{q}\n{response_text}"
        # Optionally average over steps
        total_loss = total_loss / num_steps
        return total_loss

    cls.training_step_icrl_bootstrap = training_step_icrl_bootstrap
    cls.training_step_icrl_reward_model = training_step_icrl_reward_model
    cls.training_step_icrl_ppo = training_step_icrl_ppo 