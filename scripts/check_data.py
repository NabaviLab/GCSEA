import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsea.data import SlideBags
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);args=p.parse_args()
ds=SlideBags(args.manifest)
for i in range(len(ds)):
    s=ds[i];assert s['patches'].ndim==4 and s['patches'].shape[1]==3
    assert 0<=s['label']<4
    print(s['slide_id'],s['resnet'].shape,s['uni'].shape,s['patches'].shape,s['coords'].shape)
