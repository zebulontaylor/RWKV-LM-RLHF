########################################################################################################
# ICRL H5 Generator - Convert ICRL JSONL to H5 format for training
########################################################################################################

import json
import h5py
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add tokenizer path
sys.path.append('tokenizer')
sys.path.append('tokenizer/world')

def load_tokenizer(tokenizer_path: str):
    """Load the appropriate tokenizer"""
    try:
        if 'world' in tokenizer_path:
            from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
            # Fix: Point to the actual vocabulary file in the world directory
            vocab_file = os.path.join(tokenizer_path, 'rwkv_vocab_v20230424.txt')
            if os.path.exists(vocab_file):
                return TRIE_TOKENIZER(vocab_file)
            else:
                # Fallback to the main tokenizer directory
                vocab_file = 'tokenizer/rwkv_vocab_v20230424.txt'
                return TRIE_TOKENIZER(vocab_file)
        else:
            # Try other tokenizer types as fallback
            from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
            return TRIE_TOKENIZER('tokenizer/rwkv_vocab_v20230424.txt')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using default world tokenizer")
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        # Fix: Use the actual vocabulary file path
        return TRIE_TOKENIZER('tokenizer/rwkv_vocab_v20230424.txt')


def convert_icrl_jsonl_to_h5(input_jsonl: str, output_h5: str, tokenizer_path: str, 
                           ctx_len: int = 4096, k_exemplars: int = 3, n_queries: int = 1):
    """
    Convert ICRL JSONL data to H5 format for training
    """
    print(f"Converting {input_jsonl} to H5 format...")
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Read JSONL data
    episodes = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line.strip()))
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Create H5 file
    with h5py.File(output_h5, 'w') as h5f:
        
        processed_count = 0
        
        for episode in episodes:
            try:
                episode_id = episode.get('episode_id', f'episode_{processed_count}')
                
                # Process exemplars
                exemplar_text = ""
                exemplars = episode.get('exemplars', [])[:k_exemplars]
                
                for exemplar in exemplars:
                    prompt = exemplar.get('prompt', '')
                    answer = exemplar.get('answer', '')
                    exemplar_text += f"{prompt}\n{answer}\n\n"
                
                # Process queries
                query_text = ""
                preferred_text = ""
                rejected_text = ""
                
                queries = episode.get('queries', [])[:n_queries]
                
                for query in queries:
                    prompt = query.get('prompt', '')
                    preferred = query.get('preferred_answer', '')
                    rejected = query.get('rejected_answer', '')
                    
                    query_text += f"{prompt}\n"
                    preferred_text += f"{preferred}\n"
                    if rejected:
                        rejected_text += f"{rejected}\n"
                
                # Tokenize
                try:
                    exemplar_tokens = tokenizer.encode(exemplar_text)
                    query_tokens = tokenizer.encode(query_text)
                    preferred_tokens = tokenizer.encode(preferred_text)
                    rejected_tokens = tokenizer.encode(rejected_text) if rejected_text else []
                except Exception as e:
                    print(f"Tokenization error for episode {episode_id}: {e}")
                    continue
                
                # Ensure episode fits in context
                total_length = len(exemplar_tokens) + len(query_tokens) + len(preferred_tokens)
                if total_length > ctx_len:
                    # Truncate exemplars if needed
                    max_exemplar_len = ctx_len - len(query_tokens) - len(preferred_tokens) - 50
                    if max_exemplar_len > 0:
                        exemplar_tokens = exemplar_tokens[:max_exemplar_len]
                    else:
                        print(f"Episode {episode_id} too long, skipping")
                        continue
                
                # Create episode group in H5
                episode_group = h5f.create_group(episode_id)
                
                # Store tokenized data
                episode_group.create_dataset('exemplar_tokens', data=np.array(exemplar_tokens, dtype=np.int32))
                episode_group.create_dataset('query_tokens', data=np.array(query_tokens, dtype=np.int32))
                episode_group.create_dataset('preferred_tokens', data=np.array(preferred_tokens, dtype=np.int32))
                
                if rejected_tokens:
                    episode_group.create_dataset('rejected_tokens', data=np.array(rejected_tokens, dtype=np.int32))
                
                # Store metadata
                episode_group.attrs['episode_id'] = episode_id
                episode_group.attrs['task_type'] = episode.get('task_type', 'general')
                episode_group.attrs['k_exemplars'] = len(exemplars)
                episode_group.attrs['n_queries'] = len(queries)
                episode_group.attrs['total_length'] = total_length
                
                # Store original text for debugging
                episode_group.attrs['exemplar_text'] = exemplar_text
                episode_group.attrs['query_text'] = query_text
                episode_group.attrs['preferred_text'] = preferred_text
                if rejected_text:
                    episode_group.attrs['rejected_text'] = rejected_text
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} episodes...")
                    
            except Exception as e:
                print(f"Error processing episode {episode.get('episode_id', 'unknown')}: {e}")
                continue
    
    print(f"Converted {processed_count} episodes to H5 format")
    print(f"Saved to {output_h5}")


def main():
    parser = argparse.ArgumentParser(description="Convert ICRL JSONL to H5 format")
    parser.add_argument("--input_jsonl", type=str, required=True, 
                       help="Input JSONL file path")
    parser.add_argument("--output_h5", type=str, required=True, 
                       help="Output H5 file path")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/world", 
                       help="Tokenizer path")
    parser.add_argument("--ctx_len", type=int, default=4096, 
                       help="Maximum context length")
    parser.add_argument("--k_exemplars", type=int, default=3, 
                       help="Number of exemplars per episode")
    parser.add_argument("--n_queries", type=int, default=1, 
                       help="Number of queries per episode")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output_h5).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check input file exists
    if not Path(args.input_jsonl).exists():
        print(f"Error: Input file {args.input_jsonl} does not exist")
        return
    
    # Convert to H5
    convert_icrl_jsonl_to_h5(
        args.input_jsonl,
        args.output_h5,
        args.tokenizer_path,
        args.ctx_len,
        args.k_exemplars,
        args.n_queries
    )


if __name__ == "__main__":
    main() 