########################################################################################################
# ICRL (In-Context Reinforcement Learning) Implementation
########################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy


def training_step_icrl_bootstrap(self, batch, batch_idx):
    """
    ICRL Bootstrap Training (Stage 1)
    Short training run on mixed demonstration→query corpus with KL penalty
    """
    args = self.args
    
    # Standard SFT loss on episode context + preferred answers
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    # Forward pass
    logits, _ = self(input_ids, None, full_output=True)
    
    # Compute SFT loss
    sft_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    # KL penalty to base model if available
    kl_loss = 0.0
    if hasattr(self, 'base_model') and self.base_model is not None:
        with torch.no_grad():
            base_logits, _ = self.base_model(input_ids, None, full_output=True)
        
        # Compute KL divergence
        log_p_policy = F.log_softmax(logits, dim=-1)
        p_base = F.softmax(base_logits, dim=-1)
        kl_loss = F.kl_div(log_p_policy, p_base, reduction='mean')
    
    # Total loss
    total_loss = sft_loss + args.sft_kl_alpha * kl_loss
    
    # Log metrics
    self.log("bootstrap_sft_loss", sft_loss, prog_bar=True)
    self.log("bootstrap_kl_loss", kl_loss, prog_bar=True)
    self.log("bootstrap_total_loss", total_loss, prog_bar=True)
    
    return total_loss


def training_step_icrl_reward_model(self, batch, batch_idx):
    """
    ICRL Reward Model Training (Stage 2)
    Train value head on pairwise preference data
    """
    args = self.args
    bsz = len(batch['context_tokens'])
    
    total_loss = 0.0
    correct_preferences = 0
    
    for i in range(bsz):
        context_tokens = batch['context_tokens'][i].unsqueeze(0).to(self.device)
        preferred_tokens = batch['preferred_tokens'][i].unsqueeze(0).to(self.device)
        rejected_tokens = batch['rejected_tokens'][i].unsqueeze(0).to(self.device)
        
        # Combine context with preferred/rejected answers
        preferred_input = torch.cat([context_tokens, preferred_tokens], dim=1)
        rejected_input = torch.cat([context_tokens, rejected_tokens], dim=1)
        
        # Ensure inputs fit in context
        if preferred_input.size(1) > args.ctx_len:
            preferred_input = preferred_input[:, :args.ctx_len]
        if rejected_input.size(1) > args.ctx_len:
            rejected_input = rejected_input[:, :args.ctx_len]
        
        # Forward through model to get final hidden states
        with torch.no_grad():
            _, preferred_hidden = self(preferred_input, None, full_output=False)
            _, rejected_hidden = self(rejected_input, None, full_output=False)
        
        # Get value predictions from value head
        preferred_value = self.value_head(preferred_hidden[:, -1, :])  # Last token
        rejected_value = self.value_head(rejected_hidden[:, -1, :])
        
        # Preference loss: preferred should have higher value
        preference_loss = -F.logsigmoid(preferred_value - rejected_value)
        total_loss += preference_loss
        
        # Track accuracy
        if preferred_value > rejected_value:
            correct_preferences += 1
    
    total_loss = total_loss / bsz
    accuracy = correct_preferences / bsz
    
    # Log metrics
    self.log("reward_model_loss", total_loss, prog_bar=True)
    self.log("reward_model_accuracy", accuracy, prog_bar=True)
    
    return total_loss


def training_step_icrl_ppo(self, batch, batch_idx):
    """
    ICRL PPO Training (Stage 3-4)
    Main ICRL training with PPO and exploration fixes
    """
    args = self.args
    bsz = len(batch['context_tokens'])
    
    if bsz != 1:
        raise ValueError("ICRL PPO currently supports batch size 1 only")
    
    # Get episode data
    context_tokens = batch['context_tokens'][0].to(self.device)
    preferred_tokens = batch['preferred_tokens'][0].to(self.device)
    episode_id = batch['episode_ids'][0]
    
    # Generate multiple completions for the episode
    gen_count = args.icrl_gen_count
    generated_episodes = []
    
    # Prepare prompt (context)
    prompt_tokens = context_tokens.unsqueeze(0)  # [1, seq_len]
    
    for g_i in range(gen_count):
        # Apply epsilon-mixture: occasionally sample from base model
        use_base_model = random.random() < args.icrl_epsilon_mixture
        
        if use_base_model and hasattr(self, 'base_model'):
            gen_tokens = self._generate_icrl_completion(
                self.base_model, prompt_tokens, args.icrl_gen_length,
                args.icrl_gen_temperature, args.icrl_gen_topp
            )
        else:
            gen_tokens = self._generate_icrl_completion(
                self, prompt_tokens, args.icrl_gen_length,
                args.icrl_gen_temperature, args.icrl_gen_topp
            )
        
        # Combine prompt and generation
        full_episode = torch.cat([prompt_tokens, gen_tokens], dim=1)
        if full_episode.size(1) > args.ctx_len:
            full_episode = full_episode[:, :args.ctx_len]
            
        generated_episodes.append(full_episode)
    
    # Compute rewards for generated episodes
    rewards = []
    for episode in generated_episodes:
        if hasattr(self, 'reward_model') and self.reward_model is not None:
            reward = self._compute_episode_reward(episode)
        else:
            # Fallback: simple rule-based reward
            reward = self._compute_rule_based_reward(episode, preferred_tokens)
        rewards.append(reward)
    
    # Normalize rewards (advantages)
    rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std() + 1e-8
    advantages = (rewards_tensor - mean_reward) / std_reward
    
    # Entropy bonus on hidden state norm (exploration fix)
    entropy_bonuses = []
    for episode in generated_episodes:
        _, hidden_states = self(episode, None, full_output=False)
        hidden_norm = torch.norm(hidden_states, dim=-1).mean()
        entropy_bonus = args.icrl_entropy_beta * torch.log(hidden_norm + 1e-8)
        entropy_bonuses.append(entropy_bonus)
    
    # PPO updates
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    kl_loss = 0.0
    
    # Pad all episodes to same length for batch processing
    max_len = max(ep.size(1) for ep in generated_episodes)
    padded_episodes = []
    for ep in generated_episodes:
        if ep.size(1) < max_len:
            padding = torch.zeros(1, max_len - ep.size(1), dtype=torch.long, device=self.device)
            ep = torch.cat([ep, padding], dim=1)
        padded_episodes.append(ep)
    
    batch_episodes = torch.cat(padded_episodes, dim=0)  # [gen_count, max_len]
    
    # Get current policy logits and values
    current_logits, current_hidden = self(batch_episodes, None, full_output=True)
    
    # Get reference (old) policy logits
    with torch.no_grad():
        if hasattr(self, 'reference_model') and self.reference_model is not None:
            ref_logits, _ = self.reference_model(batch_episodes, None, full_output=True)
        else:
            ref_logits = current_logits.detach()
    
    # Value predictions
    if args.icrl_value_head and hasattr(self, 'value_head'):
        values = self.value_head(current_hidden[:, -1, :])  # [gen_count, 1]
        values = values.squeeze(-1)  # [gen_count]
    else:
        values = torch.zeros_like(advantages)
    
    # PPO loss computation
    for ppo_epoch in range(args.icrl_ppo_epochs):
        for i, (episode, advantage, entropy_bonus) in enumerate(zip(generated_episodes, advantages, entropy_bonuses)):
            
            gen_length = episode.size(1) - prompt_tokens.size(1)
            if gen_length <= 0:
                continue
                
            # Get tokens for generated part only
            gen_tokens = episode[0, prompt_tokens.size(1):]
            gen_tokens = gen_tokens[:gen_length]
            
            # Get logprobs for generated tokens
            episode_logits = current_logits[i, prompt_tokens.size(1):prompt_tokens.size(1)+gen_length]
            episode_ref_logits = ref_logits[i, prompt_tokens.size(1):prompt_tokens.size(1)+gen_length]
            
            # Current policy log probabilities
            log_probs = F.log_softmax(episode_logits, dim=-1)
            current_log_probs = log_probs[torch.arange(len(gen_tokens)), gen_tokens]
            
            # Reference policy log probabilities
            ref_log_probs = F.log_softmax(episode_ref_logits, dim=-1)
            ref_log_probs_tokens = ref_log_probs[torch.arange(len(gen_tokens)), gen_tokens]
            
            # Importance sampling ratio
            ratio = torch.exp(current_log_probs.sum() - ref_log_probs_tokens.sum())
            
            # Clipped surrogate objective
            clipped_ratio = torch.clamp(ratio, 1.0 - args.icrl_clip_ratio, 1.0 + args.icrl_clip_ratio)
            policy_objective = torch.min(ratio * advantage, clipped_ratio * advantage)
            
            # Policy loss (negative because we want to maximize)
            policy_loss_i = -policy_objective - entropy_bonus
            policy_loss += policy_loss_i / (gen_count * args.icrl_ppo_epochs)
            
            # Value loss
            if args.icrl_value_head:
                value_target = advantage + mean_reward  # Simple target
                value_loss_i = F.mse_loss(values[i], value_target.detach())
                value_loss += value_loss_i / (gen_count * args.icrl_ppo_epochs)
            
            # KL divergence penalty
            kl_div = F.kl_div(
                current_log_probs, 
                F.softmax(episode_ref_logits, dim=-1), 
                reduction='sum'
            )
            kl_loss += args.icrl_kl_beta * kl_div / (gen_count * args.icrl_ppo_epochs)
    
    # Total loss
    total_loss = policy_loss + args.icrl_value_loss_coef * value_loss + kl_loss
    
    # Log metrics
    self.log("icrl_policy_loss", policy_loss, prog_bar=True)
    self.log("icrl_value_loss", value_loss, prog_bar=True)
    self.log("icrl_kl_loss", kl_loss, prog_bar=True)
    self.log("icrl_mean_reward", mean_reward, prog_bar=True)
    self.log("icrl_total_loss", total_loss, prog_bar=True)
    
    return total_loss


def _generate_icrl_completion(self, model, prompt_tokens, max_length, temperature, top_p):
    """Generate completion for ICRL episode"""
    model.eval()
    
    with torch.no_grad():
        generated = prompt_tokens.clone()
        
        for _ in range(max_length):
            if generated.size(1) >= self.args.ctx_len:
                break
                
            logits, _ = model(generated, None, full_output=True)
            next_token_logits = logits[0, -1, :]  # Last token logits
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens (simple stopping criterion)
            if next_token.item() in [0, 1, 2]:  # Common stop tokens
                break
                
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    model.train()
    return generated[:, prompt_tokens.size(1):]  # Return only generated part


def _compute_episode_reward(self, episode):
    """Compute reward for an episode using the reward model"""
    if not hasattr(self, 'reward_model') or self.reward_model is None:
        return 0.0
        
    with torch.no_grad():
        _, hidden_states = self.reward_model(episode, None, full_output=False)
        reward = self.reward_model.value_head(hidden_states[:, -1, :])
        return float(reward.squeeze())


def _compute_rule_based_reward(self, episode, preferred_tokens):
    """Compute rule-based reward as fallback"""
    # Simple similarity-based reward
    episode_tokens = episode[0, :]
    
    # Check for exact match with preferred answer
    if len(episode_tokens) >= len(preferred_tokens):
        for i in range(len(episode_tokens) - len(preferred_tokens) + 1):
            if torch.equal(episode_tokens[i:i+len(preferred_tokens)], preferred_tokens):
                return 1.0
    
    # Compute token overlap reward
    episode_set = set(episode_tokens.tolist())
    preferred_set = set(preferred_tokens.tolist())
    
    if len(preferred_set) == 0:
        return 0.0
        
    overlap = len(episode_set.intersection(preferred_set))
    return float(overlap) / len(preferred_set)


def add_icrl_value_head(model, args):
    """Add value head to model for ICRL"""
    if args.icrl_value_head and not hasattr(model, 'value_head'):
        # Add value head as a linear layer
        hidden_size = model.args.n_embd
        model.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize value head weights
        nn.init.zeros_(model.value_head.weight)
        nn.init.zeros_(model.value_head.bias)
        
        # Move to device
        model.value_head = model.value_head.to(model.device)
        
        print(f"Added ICRL value head: {hidden_size} -> 1")


def setup_icrl_models(model, args):
    """Setup reference and reward models for ICRL"""
    # Create reference model (copy of current model)
    if args.icrl and not hasattr(model, 'reference_model'):
        model.reference_model = copy.deepcopy(model)
        model.reference_model.eval()
        
        # Freeze reference model
        for param in model.reference_model.parameters():
            param.requires_grad = False
            
        print("Created ICRL reference model")
    
    # Load reward model if specified
    if args.icrl_reward_model_path and not hasattr(model, 'reward_model'):
        print(f"Loading ICRL reward model from {args.icrl_reward_model_path}")
        reward_model = copy.deepcopy(model)
        
        # Load reward model weights
        checkpoint = torch.load(args.icrl_reward_model_path, map_location=model.device)
        reward_model.load_state_dict(checkpoint, strict=False)
        reward_model.eval()
        
        # Freeze reward model
        for param in reward_model.parameters():
            param.requires_grad = False
            
        model.reward_model = reward_model
        print("Loaded ICRL reward model")


# Patch the methods into the model class
def patch_icrl_methods(model_class):
    """Patch ICRL methods into the model class"""
    model_class.training_step_icrl_bootstrap = training_step_icrl_bootstrap
    model_class.training_step_icrl_reward_model = training_step_icrl_reward_model
    model_class.training_step_icrl_ppo = training_step_icrl_ppo
    model_class._generate_icrl_completion = _generate_icrl_completion
    model_class._compute_episode_reward = _compute_episode_reward
    model_class._compute_rule_based_reward = _compute_rule_based_reward 