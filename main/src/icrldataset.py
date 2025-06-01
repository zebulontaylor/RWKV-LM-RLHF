########################################################################################################
# ICRL Dataset - In-Context Reinforcement Learning
########################################################################################################

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import h5py

class ICRLDataset(Dataset):
    """
    Dataset for In-Context Reinforcement Learning (ICRL)
    
    Handles episodes structured as "k exemplars + n query-answer pairs"
    Supports both JSONL and H5 formats
    """
    
    def __init__(self, args, file_path: str, ctx_len: int = 4096):
        self.args = args
        self.ctx_len = ctx_len
        self.k_exemplars = args.icrl_k_exemplars
        self.n_queries = args.icrl_n_queries
        self.episode_length = args.icrl_episode_length
        
        self.episodes = []
        
        if file_path.endswith('.jsonl'):
            self._load_jsonl(file_path)
        elif file_path.endswith('.h5'):
            self._load_h5(file_path)
        else:
            raise ValueError("Unsupported file format. Use .jsonl or .h5")
            
        print(f"Loaded {len(self.episodes)} ICRL episodes")
        
    def _load_jsonl(self, file_path: str):
        """Load ICRL data from JSONL format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if self._validate_episode(data):
                    self.episodes.append(data)
                    
    def _load_h5(self, file_path: str):
        """Load ICRL data from H5 format"""
        with h5py.File(file_path, 'r') as f:
            for episode_id in f.keys():
                episode_data = {
                    'episode_id': episode_id,
                    'exemplar_tokens': f[episode_id]['exemplar_tokens'][:],
                    'query_tokens': f[episode_id]['query_tokens'][:],
                    'preferred_tokens': f[episode_id]['preferred_tokens'][:],
                    'rejected_tokens': f[episode_id]['rejected_tokens'][:] if 'rejected_tokens' in f[episode_id] else None,
                    'reward_scores': f[episode_id]['reward_scores'][:] if 'reward_scores' in f[episode_id] else None
                }
                self.episodes.append(episode_data)
                
    def _validate_episode(self, episode: Dict[str, Any]) -> bool:
        """Validate that episode has required structure"""
        required_fields = ['exemplars', 'queries', 'episode_id']
        
        if not all(field in episode for field in required_fields):
            return False
            
        if len(episode['exemplars']) < 1 or len(episode['queries']) < 1:
            return False
            
        return True
        
    def _create_episode_context(self, episode: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create episode context with k exemplars + n queries
        Returns: (context_tokens, preferred_tokens, rejected_tokens)
        """
        if 'exemplar_tokens' in episode:
            # H5 format - tokens already processed
            exemplar_tokens = torch.tensor(episode['exemplar_tokens'], dtype=torch.long)
            query_tokens = torch.tensor(episode['query_tokens'], dtype=torch.long)
            preferred_tokens = torch.tensor(episode['preferred_tokens'], dtype=torch.long)
            rejected_tokens = torch.tensor(episode['rejected_tokens'], dtype=torch.long) if episode.get('rejected_tokens') is not None else None
            
            # Combine exemplars and query
            context_tokens = torch.cat([exemplar_tokens, query_tokens])
            
        else:
            # JSONL format - need to tokenize
            from ..tokenizer.world import TRIE_TOKENIZER
            tokenizer = TRIE_TOKENIZER('tokenizer/world')
            
            # Build context with exemplars
            context_text = ""
            exemplars = episode['exemplars'][:self.k_exemplars]  # Take k exemplars
            
            for exemplar in exemplars:
                context_text += f"{exemplar['prompt']}\n{exemplar['answer']}\n\n"
                
            # Add queries
            queries = episode['queries'][:self.n_queries]  # Take n queries
            query_text = ""
            preferred_text = ""
            rejected_text = ""
            
            for query in queries:
                query_text += f"{query['prompt']}\n"
                preferred_text += f"{query['preferred_answer']}\n"
                if 'rejected_answer' in query:
                    rejected_text += f"{query['rejected_answer']}\n"
                    
            # Tokenize
            context_tokens = torch.tensor(tokenizer.encode(context_text + query_text), dtype=torch.long)
            preferred_tokens = torch.tensor(tokenizer.encode(preferred_text), dtype=torch.long)
            rejected_tokens = torch.tensor(tokenizer.encode(rejected_text), dtype=torch.long) if rejected_text else None
            
        # Truncate to episode length
        max_context_len = self.episode_length - len(preferred_tokens) - 10  # Leave space for answers
        if len(context_tokens) > max_context_len:
            context_tokens = context_tokens[:max_context_len]
            
        return context_tokens, preferred_tokens, rejected_tokens
        
    def _apply_episodic_randomization(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Apply episodic randomization to prevent overfitting to task order"""
        if random.random() < 0.3:  # 30% chance of randomization
            if 'exemplars' in episode:
                # Shuffle exemplars
                episode = episode.copy()
                episode['exemplars'] = episode['exemplars'].copy()
                random.shuffle(episode['exemplars'])
                
                # Randomly shuffle queries
                episode['queries'] = episode['queries'].copy()
                random.shuffle(episode['queries'])
                
        return episode
        
    def __len__(self):
        return len(self.episodes)
        
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # Apply episodic randomization
        episode = self._apply_episodic_randomization(episode)
        
        # Create episode context
        context_tokens, preferred_tokens, rejected_tokens = self._create_episode_context(episode)
        
        # Prepare data for different ICRL stages
        if self.args.icrl_bootstrap:
            # Bootstrap stage: return context + preferred for SFT-style training
            full_tokens = torch.cat([context_tokens, preferred_tokens])
            if len(full_tokens) > self.ctx_len:
                full_tokens = full_tokens[:self.ctx_len]
                
            return {
                'input_ids': full_tokens[:-1],
                'labels': full_tokens[1:],
                'episode_id': episode.get('episode_id', f'episode_{idx}'),
                'context_length': len(context_tokens),
                'stage': 'bootstrap'
            }
            
        elif self.args.icrl_reward_model:
            # Reward model stage: return pairs for preference learning
            item = {
                'context_tokens': context_tokens,
                'preferred_tokens': preferred_tokens,
                'rejected_tokens': rejected_tokens,
                'episode_id': episode.get('episode_id', f'episode_{idx}'),
                'stage': 'reward_model'
            }
            
            if rejected_tokens is None:
                # Generate negative example by corrupting preferred
                rejected_tokens = self._generate_negative_example(preferred_tokens)
                
            item['rejected_tokens'] = rejected_tokens
            return item
            
        elif self.args.icrl:
            # PPO stage: return context for episode-based RL
            return {
                'context_tokens': context_tokens,
                'preferred_tokens': preferred_tokens,
                'episode_id': episode.get('episode_id', f'episode_{idx}'),
                'reward_scores': episode.get('reward_scores', None),
                'k_exemplars': len(episode.get('exemplars', [])),
                'n_queries': len(episode.get('queries', [])),
                'stage': 'ppo'
            }
        else:
            # Default: return as regular training data
            full_tokens = torch.cat([context_tokens, preferred_tokens])
            if len(full_tokens) > self.ctx_len:
                full_tokens = full_tokens[:self.ctx_len]
                
            return {
                'input_ids': full_tokens[:-1],
                'labels': full_tokens[1:],
                'episode_id': episode.get('episode_id', f'episode_{idx}')
            }
            
    def _generate_negative_example(self, preferred_tokens: torch.Tensor) -> torch.Tensor:
        """Generate a negative example by corrupting the preferred answer"""
        # Simple corruption: add noise, shuffle, or truncate
        corrupted = preferred_tokens.clone()
        
        corruption_type = random.choice(['noise', 'shuffle', 'truncate'])
        
        if corruption_type == 'noise' and len(corrupted) > 2:
            # Add random tokens
            noise_positions = random.sample(range(len(corrupted)), min(3, len(corrupted) // 2))
            for pos in noise_positions:
                corrupted[pos] = random.randint(0, 50000)  # Random token
                
        elif corruption_type == 'shuffle' and len(corrupted) > 4:
            # Shuffle middle portion
            mid_start = len(corrupted) // 4
            mid_end = 3 * len(corrupted) // 4
            middle = corrupted[mid_start:mid_end]
            indices = torch.randperm(len(middle))
            corrupted[mid_start:mid_end] = middle[indices]
            
        elif corruption_type == 'truncate' and len(corrupted) > 3:
            # Truncate randomly
            new_len = random.randint(len(corrupted) // 2, len(corrupted) - 1)
            corrupted = corrupted[:new_len]
            
        return corrupted

def collate_icrl_batch(batch):
    """Custom collate function for ICRL data"""
    if not batch:
        return {}
        
    stage = batch[0].get('stage', 'default')
    
    if stage == 'bootstrap':
        # Standard SFT collation
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        labels = []
        
        for item in batch:
            input_len = len(item['input_ids'])
            # Pad to max length
            padded_input = torch.zeros(max_len, dtype=torch.long)
            padded_labels = torch.full((max_len,), -100, dtype=torch.long)
            
            padded_input[:input_len] = item['input_ids']
            padded_labels[:input_len] = item['labels']
            
            input_ids.append(padded_input)
            labels.append(padded_labels)
            
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'episode_ids': [item['episode_id'] for item in batch],
            'context_lengths': [item['context_length'] for item in batch],
            'stage': stage
        }
        
    elif stage == 'reward_model':
        # Preference learning collation
        return {
            'context_tokens': [item['context_tokens'] for item in batch],
            'preferred_tokens': [item['preferred_tokens'] for item in batch],
            'rejected_tokens': [item['rejected_tokens'] for item in batch],
            'episode_ids': [item['episode_id'] for item in batch],
            'stage': stage
        }
        
    elif stage == 'ppo':
        # PPO collation
        return {
            'context_tokens': [item['context_tokens'] for item in batch],
            'preferred_tokens': [item['preferred_tokens'] for item in batch],
            'episode_ids': [item['episode_id'] for item in batch],
            'reward_scores': [item.get('reward_scores') for item in batch],
            'k_exemplars': [item['k_exemplars'] for item in batch],
            'n_queries': [item['n_queries'] for item in batch],
            'stage': stage
        }
        
    else:
        # Default collation
        return batch 