from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

class SlideBags(Dataset):
    """Manifest columns: slide_id,path,label,fold(optional). Every .pt is a slide bag."""
    def __init__(self, manifest, max_patches=None, train=False):
        self.rows = pd.read_csv(manifest).to_dict('records')
        self.max_patches = max_patches
        self.train = train
        if not self.rows: raise ValueError('Empty slide manifest')

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        bag = torch.load(str(row['path']), map_location='cpu', weights_only=False)
        required = ('resnet', 'uni', 'coords', 'patches')
        for k in required:
            if k not in bag: raise ValueError(f"Missing {k} from {row['path']}")
        n = bag['resnet'].shape[0]
        if not all(bag[k].shape[0] == n for k in required):
            raise ValueError(f'Unmatched patch counts in {row["path"]}')
        if n == 0: raise ValueError('Empty slide bag')
        if self.max_patches and n > self.max_patches:
            # Sampling is used only as a memory cap; the IRM still ranks the sampled candidates.
            selection = (torch.randperm(n) if self.train else torch.linspace(0,n-1,self.max_patches).long())[:self.max_patches]
        else: selection = torch.arange(n)
        return {k: torch.as_tensor(bag[k])[selection].float() for k in required} | {
            'label': torch.tensor(int(row['label']), dtype=torch.long), 'slide_id': str(row['slide_id'])}

def slide_collate(batch):
    # Variable-length bags. A list avoids incorrect zero-padding/top-M masking.
    return batch
