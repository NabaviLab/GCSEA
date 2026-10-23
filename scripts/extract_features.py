"""Extract ResNet50 and UNI features from already-extracted image patches.
UNI requires an authentic local checkpoint or a specified valid timm pretrained model identifier.
No random foundation-model feature vectors are accepted as research results.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader

class PatchDataset(Dataset):
    def __init__(self,df,size):self.df=df.reset_index(drop=True);self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        with Image.open(self.df.iloc[i].patch_path) as im: return self.tf(im.convert('RGB'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--patch-csv',required=True,help='Columns slide_id,patch_path,x,y,label')
    p.add_argument('--out-dir',required=True);p.add_argument('--uni-model',required=True,help='timm model name for authentic UNI checkpoint')
    p.add_argument('--uni-checkpoint',help='state_dict checkpoint for UNI-compatible architecture')
    p.add_argument('--resnet-checkpoint',help='state_dict for user trained ResNet50; otherwise torchvision ImageNet weights')
    p.add_argument('--size',type=int,default=224);p.add_argument('--batch-size',type=int,default=16);args=p.parse_args()
    try: import timm
    except ImportError as e: raise RuntimeError('Install timm to extract UNI features') from e
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if not args.resnet_checkpoint else None)
    if args.resnet_checkpoint:res.load_state_dict(torch.load(args.resnet_checkpoint,map_location='cpu',weights_only=False))
    res.fc=torch.nn.Identity();res.eval().to(device)
    uni=timm.create_model(args.uni_model,pretrained=(args.uni_checkpoint is None),num_classes=0)
    if args.uni_checkpoint:
        sd=torch.load(args.uni_checkpoint,map_location='cpu',weights_only=False)
        if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
        uni.load_state_dict(sd,strict=True)
    uni.eval().to(device)
    data=pd.read_csv(args.patch_csv);required={'slide_id','patch_path','x','y','label'}
    if not required.issubset(data.columns):raise ValueError(f'Missing columns {required-set(data.columns)}')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);manifest=[]
    for slide,df in data.groupby('slide_id',sort=False):
        ds=PatchDataset(df,args.size);loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False)
        h=[];u=[];patches=[]
        with torch.no_grad():
            for batch in loader:
                h.append(res(batch.to(device)).cpu());u.append(uni(batch.to(device)).cpu())
                # restore RGB [0,1] (normalization was ImageNet RGB)
                restored=(batch*torch.tensor([.229,.224,.225])[None,:,None,None]+torch.tensor([.485,.456,.406])[None,:,None,None]).clamp(0,1)
                patches.append(restored.cpu().to(torch.float16))
        file=out/(str(slide).replace('/','_')+'.pt')
        torch.save({'resnet':torch.cat(h),'uni':torch.cat(u),'patches':torch.cat(patches),
                    'coords':torch.tensor(df[['x','y']].to_numpy(dtype='float32'))},file)
        manifest.append({'slide_id':slide,'path':str(file.resolve()),'label':int(df.label.iloc[0])})
    pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False);print('Saved',out/'manifest.csv')
if __name__=='__main__':main()
