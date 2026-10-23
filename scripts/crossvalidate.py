import argparse,subprocess,sys,json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--concept-embeddings',required=True);p.add_argument('--config',default='config.yaml');p.add_argument('--out',default='runs/cv');p.add_argument('--folds',type=int,default=5);args=p.parse_args()
df=pd.read_csv(args.manifest);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
skf=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=42)
for fold,(tr,va) in enumerate(skf.split(df,df.label)):
    fd=out/f'fold_{fold}';fd.mkdir(exist_ok=True)
    df.iloc[tr].to_csv(fd/'train.csv',index=False);df.iloc[va].to_csv(fd/'val.csv',index=False)
    subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--config',args.config,'--train',str(fd/'train.csv'),'--val',str(fd/'val.csv'),'--concept-embeddings',args.concept_embeddings,'--out',str(fd)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'--checkpoint',str(fd/'best.pt'),'--manifest',str(fd/'val.csv'),'--out',str(fd/'predictions.csv')],check=True)
print('Finished folds; see each fold predictions.csv and history.json')
