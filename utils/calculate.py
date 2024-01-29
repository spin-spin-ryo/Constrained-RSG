import torch

def inverse_xy(x,y):
    dim = x.shape[0]
    device = x.device
    dtype = x.dtype
    # (I+xy^\top)^{-1}を求める.
    return torch.eye(dim,device=device,dtype=dtype) - (x.unsqueeze(1)@y.unsqueeze(0))/(1+x@y)

def get_minimum_eigenvalue(H):
    return torch.lobpcg(H,largest=False)[0][0]

def get_maximum_eigenvalue(H):
    return torch.lobpcg(H)[0][0]

def line_search(xk,func,grad,dk,alpha,beta,loss = None):
   if loss is None:
      with torch.no_grad():
         loss = func(xk)
   lr = 1
   with torch.no_grad():
      while loss.item() - func(xk + lr*dk) < -alpha*lr*grad@dk:
            lr *= beta 
   return lr

def subspace_line_search(xk,func,projected_grad,dk,Mk,alpha,beta,loss = None):
    if loss is None:
      with torch.no_grad():
         loss = func(xk)
    lr = 1
    proj_dk = Mk.transpose(0,1)@dk
    with torch.no_grad():
        while loss.item() - func(xk + lr*proj_dk) < -alpha*lr*projected_grad@dk:
                lr *= beta 
    return lr

def generate_semidefinite(dim,rank,device):
   P = torch.randn(dim,rank,device = device)
   return P@P.transpose(0,1)/dim

def generate_symmetric(dim,device):
   P = torch.randn(dim,dim,device = device)
   return (P + P.transpose(0,1))/2


def nonnegative_projection(x,t):
    y = x.detach().clone()
    y[y<0]=0
    return y

def BallProjection(x,radius = 1):
    if x@x <= radius*radius:
        return x
    else:
        return (radius)*x/torch.linalg.norm(x)

def BoxProjection(x,radius = 1):
    y = x.detach().clone()
    y[y>radius] = radius
    y[y<-radius] = -radius
    return y

def L1projection(x,radius = 1):
  if torch.linalg.norm(x,ord=1)<=radius:
    return x
  else:
    x_ = x.detach().clone()
    
    x_/=radius
    y = torch.sort(torch.abs(x_))[0]
    l = 0
    r = y.shape[0]
    while r-l > 1:
      m = int((l+r)/2)
      lam = y[m]
      z = y -lam
      index = z >0
      if torch.sum(z[index])>1:
        l = m
      else:
        r = m
    lam = (torch.sum(y[r:]) -1)/y[r:].shape[0]
    z = torch.zeros(y.shape,device = y.device,dtype = y.dtype)
    z[x_>lam] = (x_-lam)[x_ >lam]
    z[x_<-lam] =(x_+lam)[x_ <-lam]
    return z*radius
