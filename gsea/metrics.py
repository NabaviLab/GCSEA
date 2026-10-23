import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, accuracy_score

def metrics(labels,probabilities,num_classes):
    y=np.asarray(labels); p=np.asarray(probabilities)
    pred=p.argmax(1)
    out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
         'quadratic_weighted_kappa':float(cohen_kappa_score(y,pred,weights='quadratic')) if len(np.unique(y))>1 else None}
    try:
        onehot=np.eye(num_classes)[y]
        out['macro_auc_ovr']=float(roc_auc_score(onehot,p,average='macro')) if len(np.unique(y))==num_classes else None
    except ValueError: out['macro_auc_ovr']=None
    return out
