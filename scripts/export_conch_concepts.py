"""Export actual CONCH embeddings through the user-installed official CONCH package.

The CONCH API/version/checkpoint must be installed separately; no counterfeit vectors.
Uses commonly documented conch.open_clip_custom interface, which might require
adjustment for an updated upstream CONCH release.
"""
import argparse
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--concepts',default='concepts/concept_anchors.txt');p.add_argument('--out',default='concepts/conch_text_embeddings.pt')
    a=p.parse_args()
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    except ImportError as e:raise RuntimeError('Install official CONCH and confirm its API; see README') from e
    model, _ = create_model_from_pretrained('conch_ViT-B-16',checkpoint_path=a.checkpoint)
    tokenizer=get_tokenizer();model.eval()
    descriptions=[line.strip() for line in Path(a.concepts).read_text().splitlines() if line.strip()]
    with torch.no_grad():
        tokens=tokenizer(descriptions)
        embeds=model.encode_text(tokens,normalize=True).cpu()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'embeddings':embeds,'concepts':descriptions,'checkpoint':a.checkpoint},a.out)
    print('Exported',embeds.shape,'to',a.out)
if __name__=='__main__':main()
