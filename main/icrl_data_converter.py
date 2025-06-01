########################################################################################################
# ICRL Data Converter - Convert data to ICRL JSONL format
########################################################################################################

import json
import csv
import random
import argparse
from typing import List, Dict, Any
import uuid


def convert_csv_to_icrl_jsonl(input_path: str, output_path: str, k_exemplars: int = 3, 
                             n_queries: int = 1, task_type: str = "general"):
    """
    Convert CSV data to ICRL JSONL format
    
    Expected CSV format:
    - prompt: The question/prompt
    - chosen: The preferred answer
    - rejected: The rejected answer (optional)
    - exemplar_type: 'exemplar' or 'query' (optional, auto-detected if missing)
    """
    
    print(f"Converting {input_path} to ICRL format...")
    
    # Read CSV data
    data_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_rows.append(row)
    
    print(f"Loaded {len(data_rows)} rows from CSV")
    
    # Group data into exemplars and queries
    exemplars = []
    queries = []
    
    for row in data_rows:
        prompt = row.get('prompt', '')
        chosen = row.get('chosen', '')
        rejected = row.get('rejected', '')
        exemplar_type = row.get('exemplar_type', '').lower()
        
        if not prompt or not chosen:
            continue
            
        if exemplar_type == 'exemplar' or (exemplar_type == '' and len(exemplars) < k_exemplars * 5):
            # Treat as exemplar
            exemplars.append({
                'prompt': prompt.strip(),
                'answer': chosen.strip()
            })
        else:
            # Treat as query
            query_item = {
                'prompt': prompt.strip(),
                'preferred_answer': chosen.strip()
            }
            if rejected and rejected.strip():
                query_item['rejected_answer'] = rejected.strip()
            queries.append(query_item)
    
    print(f"Found {len(exemplars)} exemplars and {len(queries)} queries")
    
    # Create episodes
    episodes = []
    episode_count = 0
    
    # Shuffle data for variety
    random.shuffle(exemplars)
    random.shuffle(queries)
    
    # Create episodes by combining k exemplars with n queries
    query_idx = 0
    while query_idx < len(queries):
        episode_exemplars = []
        episode_queries = []
        
        # Select k exemplars
        for i in range(k_exemplars):
            if i < len(exemplars):
                exemplar_idx = (episode_count * k_exemplars + i) % len(exemplars)
                exemplars_copy = exemplars[exemplar_idx].copy()
                # Shave down exemplars for context efficiency
                exemplars_copy['prompt'] = shave_text(exemplars_copy['prompt'])
                exemplars_copy['answer'] = shave_text(exemplars_copy['answer'])
                episode_exemplars.append(exemplars_copy)
        
        # Select n queries
        for i in range(n_queries):
            if query_idx + i < len(queries):
                episode_queries.append(queries[query_idx + i])
        
        if episode_exemplars and episode_queries:
            episode = {
                'exemplars': episode_exemplars,
                'queries': episode_queries,
                'episode_id': f"episode_{episode_count:06d}_{str(uuid.uuid4())[:8]}",
                'task_type': task_type
            }
            episodes.append(episode)
            episode_count += 1
        
        query_idx += n_queries
    
    print(f"Created {len(episodes)} ICRL episodes")
    
    # Write JSONL output
    with open(output_path, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    print(f"Saved ICRL data to {output_path}")
    
    # Print sample episode
    if episodes:
        print("\nSample episode:")
        print(json.dumps(episodes[0], indent=2, ensure_ascii=False))


def shave_text(text: str, max_length: int = 200) -> str:
    """
    Shave down text to save context space while preserving meaning
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundaries
    sentences = text.split('. ')
    if len(sentences) > 1:
        result = ""
        for sentence in sentences:
            if len(result + sentence + '. ') <= max_length:
                result += sentence + '. '
            else:
                break
        if result:
            return result.strip()
    
    # Fallback: simple truncation with ellipsis
    return text[:max_length-3] + "..."


def create_sample_data(output_path: str):
    """Create sample ICRL data for testing"""
    
    sample_episodes = [
        {
            "exemplars": [
                {
                    "prompt": "Question: What is 2+2?",
                    "answer": "Answer: 4"
                },
                {
                    "prompt": "Question: What is the capital of France?",
                    "answer": "Answer: Paris"
                },
                {
                    "prompt": "Question: What is 10-5?",
                    "answer": "Answer: 5"
                }
            ],
            "queries": [
                {
                    "prompt": "Question: What is 5*6?",
                    "preferred_answer": "Answer: 30",
                    "rejected_answer": "Answer: 25"
                }
            ],
            "episode_id": "sample_episode_001",
            "task_type": "math"
        },
        {
            "exemplars": [
                {
                    "prompt": "Question: Name a programming language.",
                    "answer": "Answer: Python"
                },
                {
                    "prompt": "Question: What does CPU stand for?",
                    "answer": "Answer: Central Processing Unit"
                },
                {
                    "prompt": "Question: What is the file extension for Python files?",
                    "answer": "Answer: .py"
                }
            ],
            "queries": [
                {
                    "prompt": "Question: What is the most popular version control system?",
                    "preferred_answer": "Answer: Git",
                    "rejected_answer": "Answer: SVN"
                }
            ],
            "episode_id": "sample_episode_002", 
            "task_type": "programming"
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for episode in sample_episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    print(f"Created sample ICRL data: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert data to ICRL JSONL format")
    parser.add_argument("--input_path", type=str, help="Input CSV file path")
    parser.add_argument("--output_path", type=str, default="icrl_data.jsonl", 
                       help="Output JSONL file path")
    parser.add_argument("--k_exemplars", type=int, default=3, 
                       help="Number of exemplars per episode")
    parser.add_argument("--n_queries", type=int, default=1, 
                       help="Number of queries per episode")
    parser.add_argument("--task_type", type=str, default="general", 
                       help="Task type for episodes")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample data instead of converting CSV")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.output_path)
    elif args.input_path:
        convert_csv_to_icrl_jsonl(
            args.input_path, 
            args.output_path, 
            args.k_exemplars, 
            args.n_queries, 
            args.task_type
        )
    else:
        print("Please provide --input_path or use --create_sample")
        print("\nExample usage:")
        print("python icrl_data_converter.py --input_path data.csv --output_path icrl_data.jsonl")
        print("python icrl_data_converter.py --create_sample --output_path sample_icrl_data.jsonl")


if __name__ == "__main__":
    main() 