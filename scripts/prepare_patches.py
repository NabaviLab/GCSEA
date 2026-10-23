"""OpenSlide WSI tiling with simple HSV tissue screening. Adapt QC/stain normalization for your cohort."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--slides',required=True,help='CSV slide_id,wsi_path,label')
    p.add_argument('--out-dir',required=True);p.add_argument('--size',type=int,default=224);p.add_argument('--level',type=int,default=0)
    p.add_argument('--min-tissue-fraction',type=float,default=.35);p.add_argument('--max-patches',type=int,default=2048);a=p.parse_args()
    try:import openslide
    except ImportError as e:raise RuntimeError('Install OpenSlide system library and openslide-python') from e
    rows=[];out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for r in pd.read_csv(a.slides).itertuples(index=False):
        wsi=openslide.OpenSlide(str(r.wsi_path));scale=wsi.level_downsamples[a.level]; w,h=wsi.level_dimensions[a.level]
        folder=out/str(r.slide_id);folder.mkdir(exist_ok=True)
        count=0
        for yy in range(0,h-a.size+1,a.size):
            for xx in range(0,w-a.size+1,a.size):
                if count>=a.max_patches:break
                x0,y0=int(xx*scale),int(yy*scale)
                patch=wsi.read_region((x0,y0),a.level,(a.size,a.size)).convert('RGB')
                rgb=np.asarray(patch,dtype=np.float32)/255
                tissue=((rgb.max(2)-rgb.min(2)>.08)&(rgb.mean(2)<.93)).mean()
                if tissue<a.min_tissue_fraction:continue
                path=folder/f'{x0}_{y0}.png';patch.save(path)
                rows.append({'slide_id':r.slide_id,'patch_path':str(path.resolve()),'x':x0,'y':y0,'label':int(r.label)})
                count+=1
            if count>=a.max_patches:break
        wsi.close();print(r.slide_id,count)
    pd.DataFrame(rows,columns=['slide_id','patch_path','x','y','label']).to_csv(out/'patches.csv',index=False)
if __name__=='__main__':main()
