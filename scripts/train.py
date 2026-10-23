import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from gsea.data import SlideBags,slide_collate
from gsea.model import GaborSemanticEvidence
from gsea.losses import training_loss
from gsea.metrics import metrics
from gsea.util import seed_everything,load_config,to_device

def run_epoch(net,loader,device,cfg,optimizer=None):
    train=optimizer is not None
    net.train(train)
    labels=[];probs=[];losses=[]
    if train: optimizer.zero_grad()
    for batch in loader:
        for slide in batch:
            slide=to_device(slide,device)
            with torch.set_grad_enabled(train):
                result=net(slide)
                loss,_=training_loss(result,slide['label'],cfg,device)
                if train:
                    (loss/max(1,cfg['grad_accum_steps'])).backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            labels.append(int(slide['label'].detach().cpu()))
            probs.append(torch.softmax(result['logits'].detach(),dim=-1).cpu().numpy())
            losses.append(float(loss.detach()))
        if train:
            # grad accumulation across DataLoader batches
            run_epoch.step+=1
            if run_epoch.step%max(1,cfg['grad_accum_steps'])==0:
                optimizer.step();optimizer.zero_grad()
    if train and run_epoch.step%max(1,cfg['grad_accum_steps']): optimizer.step();optimizer.zero_grad()
    return {'loss':float(np.mean(losses))}|metrics(labels,probs,cfg['num_classes'])
run_epoch.step=0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config.yaml');ap.add_argument('--train',required=True);ap.add_argument('--val',required=True)
    ap.add_argument('--concept-embeddings',required=True);ap.add_argument('--out',default='runs/default')
    args=ap.parse_args();cfg=load_config(args.config);seed_everything(cfg['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=SlideBags(args.train,cfg['max_patches_per_slide'],train=True)
    va=SlideBags(args.val,cfg['max_patches_per_slide'],train=False)
    trloader=DataLoader(tr,batch_size=1,shuffle=True,collate_fn=slide_collate,num_workers=cfg['num_workers'])
    valoader=DataLoader(va,batch_size=1,shuffle=False,collate_fn=slide_collate,num_workers=cfg['num_workers'])
    sample=tr[0]; concepts=torch.load(args.concept_embeddings,map_location='cpu',weights_only=True)
    if isinstance(concepts,dict): concepts=concepts['embeddings']
    net=GaborSemanticEvidence(sample['resnet'].shape[-1],sample['uni'].shape[-1],concepts,cfg).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf');bad=0
    for epoch in range(1,cfg['max_epochs']+1):
        run_epoch.step=0
        a=run_epoch(net,trloader,device,cfg,opt)
        b=run_epoch(net,valoader,device,cfg)
        history.append({'epoch':epoch,'train':a,'val':b})
        print(f"Epoch {epoch:03d} train loss {a['loss']:.4f} val loss {b['loss']:.4f} val F1 {b['macro_f1']:.4f}",flush=True)
        if b['loss']<best:
            best=b['loss'];bad=0
            torch.save({'model':net.state_dict(),'config':cfg,'resnet_dim':sample['resnet'].shape[-1],
                        'uni_dim':sample['uni'].shape[-1],'concept_embeddings':concepts},out/'best.pt')
        else:
            bad+=1
            if bad>=cfg['patience']:break
    (out/'history.json').write_text(json.dumps(history,indent=2))
    print('Saved',out/'best.pt')
if __name__=='__main__': main()
