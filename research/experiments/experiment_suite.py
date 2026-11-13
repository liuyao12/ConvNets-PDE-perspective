"""Small-scale reproducibility and symmetry research for Liu--Shao--Bai (2023/2024).

Usage: python experiment_suite.py --mode all --seeds 0 1
Dependencies: torch, numpy, scipy, scikit-learn.
No network, datasets, cryptography libraries or GPU required.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from numpy.polynomial import Chebyshev
from radial_activations import hardball, softball


def finite(t):
    return bool(torch.isfinite(t).all().item())


def experiment_activation():
    report = {}
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for value in (0., 1e-20, 1., 1e4):
            if dtype in (torch.float16, torch.bfloat16) and value==1e-20:
                continue
            x = torch.full((2,8,3,3), value, dtype=dtype, requires_grad=True)
            norm = torch.sqrt((x * x).sum(dim=1, keepdim=True))
            y = torch.where(norm > math.sqrt(8), math.sqrt(8)*x/norm, x)
            y.float().mean().backward()
            old = {'forward_finite':finite(y),'grad_finite':finite(x.grad),'nan_grads':int(torch.isnan(x.grad).sum())}
            old_softball = x.detach() / torch.sqrt(1+(x.detach()*x.detach()).sum(dim=1,keepdim=True)/8)
            result = {'old_hardball':old, 'old_softball':{'forward_finite':finite(old_softball),'max_output':float(old_softball.float().abs().max())}}
            for name, cls in (('hardball',hardball),('softball',softball)):
                x = torch.full((2,8,3,3),value,dtype=dtype,requires_grad=True)
                y=cls(radius2=8)(x)
                y.float().mean().backward()
                result[name]={'forward_finite':finite(y),'grad_finite':finite(x.grad),'max_output':float(y.detach().float().abs().max())}
            report[f'{str(dtype)}:{value}'] = result
    return report


class QuadraticBlock(nn.Module):
    """Minimal model of the paper's quadratic residual operation (no BatchNorm)."""
    def __init__(self, channels=12, expansion=2, dt=1., activation='none'):
        super().__init__()
        width = channels*expansion
        self.pre = nn.Conv2d(channels,2*width,1,bias=False)
        self.spatial = nn.Conv2d(width,2*width,3,padding=1,groups=width,bias=False)
        self.post = nn.Conv2d(2*width,channels,1,bias=False)
        self.dt = dt
        self.activation = {
            'none':nn.Identity(),
            'softball':softball(radius2=channels),
            'hardball':hardball(radius2=channels),
        }[activation]
    def forward(self,x):
        a,b = self.pre(x).chunk(2,dim=1)
        return self.activation(x + self.dt*self.post(self.spatial(a*b)))


class DigitsModel(nn.Module):
    def __init__(self, kind='quadratic', channels=12, blocks=4):
        super().__init__()
        self.stem=nn.Conv2d(1,channels,3,padding=1)
        self.stem_act=nn.ReLU()
        if kind=='quadratic':
            self.blocks=nn.Sequential(*[QuadraticBlock(channels,dt=1.,activation='none') for _ in range(blocks)])
        elif kind=='quadratic_softball':
            self.blocks=nn.Sequential(*[QuadraticBlock(channels,dt=1.,activation='softball') for _ in range(blocks)])
        elif kind=='quadratic_scaled':
            self.blocks=nn.Sequential(*[QuadraticBlock(channels,dt=0.125,activation='softball') for _ in range(blocks)])
        elif kind=='symmetric_flux':
            self.blocks=nn.Sequential(*[SymmetricFluxBlock(channels,rank=8,dt=0.04,alpha=1.0) for _ in range(blocks)])
        else:
            raise ValueError(kind)
        self.head=nn.Linear(channels,10)
    def forward(self,x):
        x=self.stem_act(self.stem(x))
        x=self.blocks(x)
        return self.head(x.mean(dim=(2,3)))


def random_orthogonal(n,dtype=torch.float64):
    q,r=torch.linalg.qr(torch.randn(n,n,dtype=dtype))
    return q*torch.sign(torch.diag(r)).unsqueeze(0)


def experiment_gauge():
    """Exact input/output gauge transform of a quadratic conv block, including skips."""
    torch.manual_seed(2)
    C,W=6,12
    block=QuadraticBlock(C, expansion=2,dt=0.3,activation='softball').double()
    xin=torch.randn(3,C,7,7,dtype=torch.float64)
    Qi=random_orthogonal(C)
    Qo=random_orthogonal(C)
    xrot=torch.einsum('ij,bjhw->bihw',Qi,xin)
    result_target = torch.einsum('ij,bjhw->bihw',Qo,block(xin))
    import copy
    transformed=copy.deepcopy(block)
    with torch.no_grad():
        transformed.pre.weight.copy_(torch.einsum('oihw,ij->ojhw', block.pre.weight,Qi.T))
        transformed.post.weight.copy_(torch.einsum('ij,jkhw->ikhw',Qo,block.post.weight))
    transformed.shortcut = nn.Conv2d(C,C,1,bias=False).double()
    with torch.no_grad():
        transformed.shortcut.weight.copy_((Qo@Qi.T).reshape(C,C,1,1))
    def gauged_forward(x):
        a,b=transformed.pre(x).chunk(2,dim=1)
        return transformed.activation(transformed.shortcut(x) + transformed.dt*transformed.post(transformed.spatial(a*b)))
    y=gauged_forward(xrot)
    error=float((result_target-y).abs().max())
    # Weight recovery attack on the known public final 1x1 conv in the block.
    original=block.post.weight.detach().flatten(start_dim=1)
    encoded=transformed.post.weight.detach().flatten(start_dim=1)
    q_recovered=encoded @ torch.linalg.pinv(original)
    key_err=float((q_recovered-Qo).abs().max())
    return {'max_equivariance_error_fp64':error,'known_public_weights_key_recovery_error_fp64':key_err,
            'extra_dense_skip_parameters_per_block':C*C,'extra_parameters_fraction_vs_block':(C*C)/sum(p.numel() for p in block.parameters())}


class SymmetricFluxBlock(nn.Module):
    """Two-dimensional symmetric hyperbolic model using CP-factorized cubic potential.

    The face flux is the exact average of grad Phi on the segment joining
    adjacent states, plus Lax-Friedrichs dissipation. Spatial BC are periodic.
    """
    def __init__(self,channels,rank=6,dt=0.04,alpha=1.0):
        super().__init__()
        self.vx=nn.Parameter(torch.randn(rank,channels)*0.12)
        self.vy=nn.Parameter(torch.randn(rank,channels)*0.12)
        self.lambdax=nn.Parameter(torch.ones(rank))
        self.lambday=nn.Parameter(torch.ones(rank))
        self.dt=dt
        self.alpha=alpha
    @staticmethod
    def flux_face(x,v,lamb,dim,alpha):
        right = x.roll(-1,dims=dim)
        p = torch.einsum('rc,bchw->brhw',v,x)
        p2=p.roll(-1,dims=dim)
        coeff=lamb[None,:,None,None]*(p.square()+p*p2+p2.square())/6
        f_ec=torch.einsum('rc,brhw->bchw',v,coeff)
        return f_ec-(alpha/2)*(right-x)
    def forward(self,x):
        fx=self.flux_face(x,self.vx,self.lambdax,-1,self.alpha)
        fy=self.flux_face(x,self.vy,self.lambday,-2,self.alpha)
        dx=fx-fx.roll(1,dims=-1)
        dy=fy-fy.roll(1,dims=-2)
        return x-self.dt*(dx+dy)


def experiment_entropy():
    torch.manual_seed(10)
    C,R,N=4,7,16
    v=torch.randn(R,C,dtype=torch.float64)*0.4
    lam=torch.randn(R,dtype=torch.float64)*0.3
    u=torch.randn(1,C,1,N,dtype=torch.float64)*0.5
    alpha=1.0
    f=SymmetricFluxBlock.flux_face(u,v,lam,dim=-1,alpha=alpha)
    du=-(f-f.roll(1,dims=-1))
    dot=float((u*du).sum())
    diss=float(-alpha/2*((u.roll(-1,dims=-1)-u).square()).sum())
    # Correct change of basis for ROW vectors: v' = v Q^T.
    Q = random_orthogonal(C)
    vr=v@Q.T
    ur=torch.einsum('ij,bjhw->bihw',Q,u)
    flux_rot=SymmetricFluxBlock.flux_face(ur,vr,lam,dim=-1,alpha=alpha)
    flux_expected=torch.einsum('ij,bjhw->bihw',Q,f)
    # Forward Euler test for a small step; energy should decrease with ample LF viscosity.
    next_u=u+0.02*du
    E0=float(0.5*(u*u).sum())
    E1=float(0.5*(next_u*next_u).sum())
    a = torch.einsum('r,ri,rj->ij', lam*torch.einsum('rc,c->r',v,u[0,:,0,0]), v, v)
    matrix_symmetry_error = float((a-a.T).abs().max())
    eigs=torch.linalg.eigvalsh(a)
    return {'semidiscrete_energy_derivative':dot,'exact_dissipation_formula':diss,
            'entropy_identity_error':abs(dot-diss), 'orthogonal_flux_equivariance_error':float((flux_rot-flux_expected).abs().max()),
            'forward_euler_energy_initial':E0,'forward_euler_energy_after_dt_0p02':E1,
            'energy_decreased':E1<=E0,'principal_matrix_symmetric_error':matrix_symmetry_error,'principal_matrix_eigenvalues':[float(z) for z in eigs]}


def experiment_dynamical():
    results={}
    u0=0.4
    for name,dt,activation in [('unscaled',1.,None),('scaled',0.05,None),('clipped',1.,'softball')]:
        x=torch.tensor(u0,dtype=torch.float64)
        peak=x.item()
        first_exceeds_1e6=None
        for step in range(100):
            x=x+dt*x*x
            if activation=='softball':
                x=x/torch.sqrt(1+x*x)
            if abs(float(x))>1e6 and first_exceeds_1e6 is None:
                first_exceeds_1e6=step+1
            peak=max(peak,float(x))
        results[name]={'final':float(x) if finite(x) else None,'peak':peak if math.isfinite(peak) else None,'first_step_over_1e6':first_exceeds_1e6,'finite':finite(x)}
    return results


def experiment_polynomial():
    results={}
    for S in [1.,4.,16.]:
        grid=np.linspace(0,S,20001)
        y=1/np.sqrt(1+grid)
        for degree in [3,5,7]:
            poly=Chebyshev.fit(grid,y,deg=degree,domain=[0,S]).convert(kind=np.polynomial.Polynomial)
            error = float(np.max(np.abs(poly(grid)-y)))
            # Mistargeted out-of-range inputs: measure error at 4S.
            ood=float(abs(poly(4*S)-1/math.sqrt(1+4*S)))
            # For Horner polynomial with s=||x||^2 and x*p(s): naive sequential multiplication depth = degree+2.
            results[f'domain_0_{S:g}_degree_{degree}']={'max_abs_error_on_20001_point_grid':error,
                'abs_error_at_4_times_range':ood,
                'horner_multiplicative_depth_per_activation':degree+2,
                'poly_coefficients_low_to_high':[float(a) for a in poly.coef]}
    return results


def experiment_train(seeds=(0,),epochs=8,batch_size=96,lr=0.01):
    digits=load_digits()
    X=digits.images.astype(np.float32)[:,None,:,:]/16.0
    y=digits.target.astype(np.int64)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    xtr=torch.from_numpy(Xtr); ytr=torch.from_numpy(ytr)
    xte=torch.from_numpy(Xte); yte=torch.from_numpy(yte)
    train_results=[]
    kinds=['quadratic','quadratic_softball','quadratic_scaled','symmetric_flux']
    for seed in seeds:
        torch.manual_seed(seed)
        for kind in kinds:
            rng = torch.Generator().manual_seed(123+seed)
            torch.manual_seed(seed)
            model=DigitsModel(kind=kind,channels=12,blocks=4)
            opt=torch.optim.Adam(model.parameters(),lr=lr)
            epoch_records=[]
            overflow=False
            t0=time.perf_counter()
            for epoch in range(epochs):
                model.train()
                order=torch.randperm(len(xtr),generator=rng)
                losses=[]; max_grads=[]; max_activations=[]
                for ids in order.split(batch_size):
                    opt.zero_grad(set_to_none=True)
                    out=model(xtr[ids]); loss=F.cross_entropy(out,ytr[ids])
                    if not finite(loss): overflow=True; break
                    loss.backward()
                    grads=[p.grad for p in model.parameters() if p.grad is not None]
                    if any(not finite(g) for g in grads): overflow=True; break
                    max_grads.append(max(float(g.detach().abs().max()) for g in grads))
                    opt.step()
                    losses.append(float(loss.detach()))
                if overflow: break
                model.eval()
                with torch.no_grad():
                    logits=torch.cat([model(batch) for batch in xte.split(batch_size)])
                    accuracy=float((logits.argmax(1)==yte).float().mean())
                    max_logit=float(logits.abs().max())
                epoch_records.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                    'val_accuracy':accuracy,'max_abs_grad':max(max_grads),'max_abs_logit':max_logit})
            train_results.append({'model':kind,'seed':seed,'failed_nonfinite':overflow,
                                  'epochs_completed':len(epoch_records),
                                  'elapsed_seconds':round(time.perf_counter()-t0,2),
                                  'last_epoch':epoch_records[-1] if epoch_records else None,
                                  'all_epochs':epoch_records})
    return {'dataset':'sklearn digits 8x8 (1797 samples, stratified 70/30 split)',
            'optimizer':f'Adam lr={lr}, batch_size={batch_size}, epochs={epochs}', 'runs':train_results}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['all','nontrain','train'],default='all')
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    ap.add_argument('--out',type=Path,default=Path('results.json'))
    args=ap.parse_args()
    torch.set_num_threads(2)
    result={'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'cpu_only':not torch.cuda.is_available(),'threads':torch.get_num_threads()},
            'date_context':'2026-11-13'}
    if args.mode in ('nontrain','all'):
        result['activation']=experiment_activation()
        result['gauge']=experiment_gauge()
        result['entropy']=experiment_entropy()
        result['dynamical']=experiment_dynamical()
        result['polynomial']=experiment_polynomial()
    if args.mode in ('train','all'):
        result['digits_training']=experiment_train(seeds=args.seeds,epochs=args.epochs)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
