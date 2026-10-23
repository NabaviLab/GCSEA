import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pandas as pd
from gsea.data import SlideBags
from gsea.model import GaborSemanticEvidence
from gsea.metrics import metrics
from gsea.util import to_device

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',default='predictions.csv');args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];ds=SlideBags(args.manifest,cfg['max_patches_per_slide'])
    model=GaborSemanticEvidence(checkpoint['resnet_dim'],checkpoint['uni_dim'],checkpoint['concept_embeddings'],cfg).to(device)
    model.load_state_dict(checkpoint['model']);model.eval();rows=[];y=[];prob=[]
    with torch.no_grad():
        for sample in ds:
            output=model(to_device(sample,device)); p=torch.softmax(output['logits'],dim=-1).cpu().tolist()
            row={'slide_id':sample['slide_id'],'label':sample['label'].item(),'prediction':int(max(range(len(p)),key=lambda j:p[j]))}
            row.update({f'prob_class_{i}':v for i,v in enumerate(p)})
            rows.append(row);y.append(row['label']);prob.append(p)
    pd.DataFrame(rows).to_csv(args.out,index=False)
    print(json.dumps(metrics(y,prob,cfg['num_classes']),indent=2));print('Predictions:',args.out)
if __name__=='__main__':main()
