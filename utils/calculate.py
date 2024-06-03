from jax import jacrev,jacfwd,jit
import jax.numpy as jnp
from jax.lax import transpose
from jax import random,jvp,grad
from environments import key
import numpy as np
import jax

def hessian(f):
    return jacfwd(jacrev(f))


def inverse_xy(x,y):
    dim = x.shape[0]
    dtype = x.dtype
    # (I+xy^\top)^{-1}を求める.
    return jnp.eye(dim,dtype=dtype) - (jnp.expand_dims(x,1)@jnp.expand_dims(y,0))/(1+x@y)

def get_minimum_eigenvalue(H):
    return jnp.min(jnp.linalg.eigvalsh(H))

def get_maximum_eigenvalue(H):
    return jnp.max(jnp.linalg.eigvalsh(H))

def line_search(xk,func,grad,dk,alpha,beta,loss = None,init_lr = 1):
  if loss is None:
    loss = func(xk)
  lr = init_lr
  while loss - func(xk + lr*dk) < -alpha*lr*grad@dk:
    lr *= beta 
    if lr < 1e-12:
      return 0
  return lr

def subspace_line_search(xk,func,projected_grad,dk,Mk,alpha,beta,loss = None,init_lr=1):
    if loss is None:
      loss = func(xk)
    lr = init_lr
    proj_dk = transpose(Mk,(1,0))@dk
    while loss - func(xk + lr*proj_dk) < -alpha*lr*projected_grad@dk:
      lr *= beta
      if lr < 1e-12:
        return 0 
    return lr

def generate_semidefinite(dim,rank):
   global key
   P = random.normal(key,(dim,rank))
   key, _ = random.split(key)
   return P@transpose(P,(1,0))/dim

def generate_symmetric(dim):
   global key
   P = random.normal(key,(dim,dim))
   key, _ = random.split(key)
   return (P + transpose(P,(1,0)))/2

def jax_randn(*args,dtype = jnp.float32):
   global key
   P = random.normal(key,args).astype(dtype)
   key, _ = random.split(key)
   return P

@jit
def nonnegative_projection(x,t):
    return jnp.maximum(x, 0)

@jit
def BallProjection(x,radius = 1):
    if x@x <= radius*radius:
        return x
    else:
        return (radius)*x/jnp.linalg.norm(x)

@jit
def BoxProjection(x,radius = 1):
    y = np.array(x.copy())
    y[y>radius] = radius
    y[y<-radius] = -radius
    return jnp.array(y)

def identity_prox(x,t):
  return x

def projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))

def projection_simplex(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
  return value * projection_unit_simplex(x / value)

def L1projection(x,radius = 1):
  return jnp.sign(x) * projection_simplex(jnp.abs(x), radius)

def get_jvp(func,x,M):
  if M is not None:
    reduced_dim = M.shape[0]
    d = np.zeros(reduced_dim,dtype = x.dtype)
    for i in range(reduced_dim):
      _,directional_derivative= jvp(func,(x,),(M[i],))
      d[i] = directional_derivative
    return jnp.array(d)
  else:
    dim = x.shape[0]
    e = np.zeros(dim,dtype=x.dtype)
    e[0] = 1
    e = jnp.array(e)
    d = np.zeros(dim,dtype = x.dtype)
    for i in range(dim):
      _,directional_derivative= jvp(func,(x,),(e,))
      d[i] = directional_derivative
      e = jnp.roll(e,1)
    return jnp.array(d)
  
def clipping_eigenvalues(B,lower,upper):
  eig_vals,eig_vecs = jnp.linalg.eigh(B)
  eig_vals = np.array(eig_vals)
  eig_vals[eig_vals< lower] = lower
  eig_vals[eig_vals > upper] = upper
  eig_vals = jnp.array(eig_vals)
  return eig_vecs@jnp.diag(eig_vals)@eig_vecs.T

# forward-over-reverse
def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

def get_hessian_with_hvp(f,x,M):
  reduced_dim = M.shape[0]
  MHM = np.zeros((reduced_dim,reduced_dim),dtype = x.dtype)
  for i in range(reduced_dim):
    Hm = hvp(f,(x,),(M[i],))
    for j in range(i,reduced_dim):
      a = jnp.dot(Hm,M[j])
      MHM[i,j] = a
      MHM[j,i] = a
  return MHM

def constant(iter):
  return 1

def inverse(iter):
  return 1/iter