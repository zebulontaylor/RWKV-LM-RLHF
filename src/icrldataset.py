import json
from torch.utils.data import Dataset

class ICRLDataset(Dataset):
    def __init__(self, args, data_file, ctx_len):
        """
        ICRL episodic dataset loader
        Args:
            args: parsed arguments
            data_file: path to dataset.json
            ctx_len: context length (not used here)
        """
        super().__init__()
        with open(data_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        points = payload.get('dataset_points', [])
        self.samples = []
        for ep in points:
            context = ep.get('context', '')
            questions = ep.get('questions', [])
            criterion = ep.get('evaluation_criterion', '')
            # Keep full episode of questions for multi-step ICL
            self.samples.append({
                'context': context,
                'questions': questions,
                'evaluation_criterion': criterion
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_icrl_batch(batch):
    # batch is a list of sample dicts
    # We return as-is for micro-batch processing
    return batch 