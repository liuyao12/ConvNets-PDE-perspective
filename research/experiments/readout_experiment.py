"""Test whether conservative flux layers are useful with spatially sensitive readouts."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from experiment_suite import DigitsModel

class ReadoutModel(DigitsModel):
    def __init__(self, kind, readout):
        super().__init__(kind='symmetric_flux' if kind=='stem_only' else kind,channels=12,blocks=4)
        self.readout=readout
        if kind == 'stem_only':
            self.blocks=nn.Identity()
        self.head=nn.Linear(12 * 64 if readout == 'flatten' else 12,10)
    def forward(self,x):
        x=self.blocks(self.stem_act(self.stem(x)))
        if self.readout=='flatten': x=x.flatten(1)
        elif self.readout=='radial_mean': x=(x*(1+3*x.square().sum(1,keepdim=True))).mean((2,3))
        else: x=x.mean((2,3))
        return self.head(x)

def train():
    torch.set_num_threads(2)
    digits=load_digits()
    X=digits.images.astype(np.float32)[:,None,:,:]/16
    y=digits.target.astype(np.int64)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,stratify=y,random_state=42)
    xtr=torch.from_numpy(Xtr);ytr=torch.from_numpy(ytr)
    xte=torch.from_numpy(Xte);yte=torch.from_numpy(yte)
    out=[]
    for seed in (0,1):
        for kind,readout in [('stem_only','mean'),('symmetric_flux','mean'),('stem_only','flatten'),('symmetric_flux','flatten'),('symmetric_flux','radial_mean')]:
            torch.manual_seed(seed)
            model=ReadoutModel(kind,readout)
            opt=torch.optim.Adam(model.parameters(),lr=.01)
            gen=torch.Generator().manual_seed(seed+123)
            t0=time.perf_counter()
            for epoch in range(10):
                model.train()
                for ids in torch.randperm(len(xtr),generator=gen).split(96):
                    opt.zero_grad()
                    logits=model(xtr[ids]); loss=F.cross_entropy(logits,ytr[ids])
                    loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                pred=torch.cat([model(a) for a in xte.split(96)]).argmax(1)
                accuracy=float((pred==yte).float().mean())
            out.append({'seed':seed,'model':kind,'readout':readout,'test_accuracy':accuracy,'seconds':time.perf_counter()-t0})
            print(seed,kind,readout,round(accuracy,4))
    Path('readout_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':train()

