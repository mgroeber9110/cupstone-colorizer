import sys, numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from PIL import Image


def render(obj, png, res=1400):
    V=[];C=[]
    for l in open(obj):
        if l.startswith('v '):
            x=[float(t) for t in l.split()[1:7]]; V.append(x[:3]); C.append(x[3:])
    V=np.array(V);C=np.array(C)
    lo=V[:,:2].min(0);hi=V[:,:2].max(0)
    w=res;h=int(res*(hi[1]-lo[1])/(hi[0]-lo[0]))
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],w),np.linspace(hi[1],lo[1],h))
    # keep topmost: sort so highest z wins in Delaunay duplicates (approx)
    f=LinearNDInterpolator(V[:,:2],np.column_stack([C,V[:,2]]))
    out=f(gx,gy)
    rgb=out[...,:3]; z=out[...,3]
    # hillshade
    dz_y,dz_x=np.gradient(np.nan_to_num(z,nan=np.nanmin(z)),(hi[1]-lo[1])/h,(hi[0]-lo[0])/w)
    n=np.dstack([-dz_x,dz_y,np.ones_like(z)]);n/=np.linalg.norm(n,axis=2,keepdims=True)
    L=np.array([-0.5,0.5,0.7]);L/=np.linalg.norm(L)
    sh=np.clip(n@L,0,1)
    img=rgb*(0.55+0.45*sh[...,None])
    from scipy.spatial import cKDTree
    d,_=cKDTree(V[:,:2]).query(np.column_stack([gx.ravel(),gy.ravel()]))
    img[(d>0.06).reshape(h,w)]=np.nan
    img=np.nan_to_num(img,nan=1.0)
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(png)
if __name__=='__main__': render(sys.argv[1],sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 1400)
