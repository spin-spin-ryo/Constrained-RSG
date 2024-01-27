import torch
from algorithms.constrained_descent_method import constrained_optimization_solver
from utils.calculate import inverse_xy
import time

class RSGLC(constrained_optimization_solver):
  def __init__(self, device="cpu", dtype=torch.float64) -> None:
    self.lk = None
    self.first_check = False
    super().__init__(device, dtype)
    self.params_key = ["eps0",
                  "delta1",
                  "eps2",
                  "dim",
                  "reduced_dim",
                  "alpha1",
                  "alpha2",
                  "beta",
                  "mode",
                  "backward"]
    
  def subspace_first_order_oracle(self,x,Mk):
    reduced_dim = Mk.shape[0]
    subspace_func = lambda d:self.f(x + Mk.transpose(0,1)@d)
    if self.backward_mode:
      d = torch.zeros(reduced_dim,requires_grad=True,device=self.device,dtype=self.dtype)
      loss_d = subspace_func(d)
      loss_d.backward()
      return d.grad
  
  def __iter_per__(self,params):
    eps0 = params["eps0"]
    delta1 = params["delta1"]
    eps2 = params["eps2"]
    dim = params["dim"]
    reduced_dim = params["reduced_dim"]
    mode = params["mode"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    beta = params["beta"]


    Mk = self.generate_matrix(dim,reduced_dim,mode)
    GkMk = self.get_active_constraints_projected_grads(Mk,eps0)
    # M_k^\top \nabla f
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk)
    d = self.__direction__(projected_grad,active_constraints_projected_grads=GkMk,delta1=delta1,eps2=eps2,dim=dim,reduced_dim=reduced_dim)
    if d is None:
      return
    if Mk is None:
      Md = d
    else:
      Md = Mk.transpose(0,1)@d
    if self.first_check:
      alpha =self.__step_size__(Md,alpha2,beta)
    else:
      alpha = self.__step_size__(Md,alpha1,beta)
    
    self.__update__(alpha*Md)
    return
    
  def __step_size__(self,direction,alpha,beta):
    with torch.no_grad():
      while not self.con.is_feasible(self.xk + alpha*direction):
        alpha *= beta
      return alpha
  
  def __direction__(self,projected_grad,active_constraints_projected_grads,delta1,eps2,dim,reduced_dim):
    if active_constraints_projected_grads is None:
      self.lk = None
      A = None
      direction1 =  -projected_grad
    else:
      b = active_constraints_projected_grads@projected_grad
      A = active_constraints_projected_grads@active_constraints_projected_grads.tranpose(0,1)
      self.lk = torch.linalg.solve(A,-b)
      direction1 = -projected_grad - active_constraints_projected_grads.transpose(0,1)@self.lk

    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        l = self.lk.clone()
        l[l>0] = 0
        l *= reduced_dim/dim
        direction2 = -torch.linalg.solve(A,l)
        return direction2
    else:
      return direction1
  
  def __clear__(self):
    self.first_check = False
    return super().__clear__()
  
  def get_active_constraints(self,constraints_grads_norm,constraints_values,eps0):
    index = constraints_values > -eps0*constraints_grads_norm  
    return index
  
  def get_projected_gradient_by_matmul(Mk,G):
    # Mk:(n,d) Gk(m,n)
    # Mk = None = identity
    if len(G) == 0:
      return None
    
    if Mk is None:
      return G
    else:
      return G@Mk 

  def get_active_constraints_projected_grads(self,Mk,eps0):
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    constraints_grads_norm = torch.linalg.norm(constraints_grads,dim = 1)
    active_constraints_index = self.get_active_constraints(constraints_grads_norm=constraints_grads_norm,
                                                           constraints_values=constraints_values,
                                                           eps0=eps0)
    active_constraints_grads = constraints_grads[active_constraints_index]
    active_constraints_projected_grads = self.get_projected_gradient_by_matmul(Mk=Mk,
                                                                               G = active_constraints_grads)
    
    return active_constraints_projected_grads

  def check_norm(self,d,delta1):
    return torch.linalg.norm(d) <= delta1
  
  def check_lambda(self,eps2):
    if self.lk is None:
      return True
    return torch.min(self.lk) > -eps2
   
  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      return torch.randn(reduced_dim,dim,device = self.device,dtype=self.dtype)/dim
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")
  
class RSGNC(RSGLC):
  def __init__(self, device="cpu", dtype=torch.float64) -> None:
    super().__init__(device, dtype)
    self.params_key = ["eps0",
                  "delta1",
                  "eps2",
                  "dim",
                  "reduced_dim",
                  "alpha1",
                  "alpha2",
                  "beta",
                  "mode",
                  "r",
                  "backward"]
      
  def __iter_per__(self,params):
    # 後でx.gradについて確認
    eps0 = params["eps0"]
    delta1 = params["delta1"]
    eps2 = params["eps2"]
    dim = params["dim"]
    reduced_dim = params["reduced_dim"]
    mode = params["mode"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    beta = params["beta"]
    r = params["r"]

    Mk = self.generate_matrix(dim,reduced_dim,mode)
    GkMk = self.get_active_constraints_projected_grads(Mk,eps0)
    # M_k^\top \nabla f
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk,reduced_dim)
    d = self.__direction__(projected_grad,active_constraints_projected_grads=GkMk,delta1=delta1,eps2=eps2,dim=dim,reduced_dim=reduced_dim,r = r)
    if d is None:
      return
    if Mk is None:
      Md = d
    else:
      Md = Mk@d
    if self.first_check:
      alpha =self.__step_size__(Md,alpha2,beta)
    else:
      alpha = self.__step_size__(Md,alpha1,beta)
    
    self.__update__(alpha*Md)
    return
  
  def __direction__(self,projected_grad,active_constraints_projected_grads,delta1,eps2,dim,reduced_dim,r):
    if active_constraints_projected_grads is None:
      self.lk = None
      A = None
      direction1 =  -projected_grad
    else:
      GMMf = active_constraints_projected_grads@projected_grad
      GMMG = active_constraints_projected_grads@active_constraints_projected_grads.tranpose(0,1)
      wk = torch.linalg.norm(active_constraints_projected_grads,dim = 1)
      GMMG_inv = torch.linalg.inv(GMMG)
      rk = r/torch.sqrt(GMMG_inv@wk@wk)
      projected_grad_norm = torch.linalg.norm(projected_grad)
      v = rk*wk/projected_grad_norm
      self.lk = -GMMG_inv@GMMf
      lk_bar = -(GMMG_inv@(inverse_xy(v,self.lk)@(GMMf - rk*projected_grad_norm*wk)))
      direction1 = -projected_grad - active_constraints_projected_grads.transpose(0,1)@lk_bar

    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        if -torch.sum(self.lk) >= eps2/2:
          s = torch.ones(self.lk.shape[0],device = self.device, dtype= self.dtype)
        else:
          s = torch.ones(self.lk.shape[0],device = self.device, dtype= self.dtype)
          negative_sum = torch.sum(self.lk[self.lk<0])
          positive_sum = torch.sum(self.lk[self.lk>0])
          s[self.lk>0] = -negative_sum/positive_sum/2
        direction2 = -eps2*reduced_dim/dim*active_constraints_projected_grads.transpose(0,1)@(GMMG_inv@s)
        return direction2
    else:
      return direction1