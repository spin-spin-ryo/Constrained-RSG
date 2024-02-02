import torch
from algorithms.constrained_descent_method import constrained_optimization_solver
from utils.calculate import inverse_xy,get_jvp
import time
from utils.logger import logger
import random
import numpy as np
from environments import DIRECTIONALDERIVATIVE,FINITEDIFFERENCE



class RSGLC(constrained_optimization_solver):
  def __init__(self, device="cpu", dtype=torch.float64) -> None:
    self.lk = None
    self.first_check = False
    self.active_num = 0
    self.grad_norm = 0
    self.reduced_dim = 0
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
    if isinstance(self.backward_mode,str):
      if self.backward_mode == DIRECTIONALDERIVATIVE:
        v = torch.zeros(reduced_dim,device = self.device,dtype = self.dtype)
        for i in range(reduced_dim):
          v[i] = get_jvp(self.f,x,Mk[i])
        return v
      elif self.backward_mode == FINITEDIFFERENCE:
        h = 1e-8
        with torch.no_grad():
          z = self.f(x)
          return torch.tensor([(self.f(x + h*Mk[i]) - z)/h for i in range(reduced_dim)],device = self.device, dtype = self.dtype)
    elif self.backward_mode:
      d = torch.zeros(reduced_dim,requires_grad=True,device=self.device,dtype=self.dtype)
      loss_d = subspace_func(d)
      loss_d.backward()
      return d.grad
     
  
  def __iter_per__(self,params):
    eps0 = params["eps0"]
    delta1 = params["delta1"]
    eps2 = params["eps2"]
    dim = params["dim"]
    mode = params["mode"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    beta = params["beta"]
    Gk = self.get_active_constraints_grads(eps0)
    while self.reduced_dim-self.active_num < 5:
      self.reduced_dim += 10
      if self.reduced_dim > dim:
        self.reduced_dim = dim
        break
    Mk = self.generate_matrix(dim,self.reduced_dim,mode)
    GkMk = self.get_projected_gradient_by_matmul(Mk,Gk)
    # M_k^\top \nabla f
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk)
    d = self.__direction__(projected_grad,active_constraints_projected_grads=GkMk,delta1=delta1,eps2=eps2,dim=dim,reduced_dim=self.reduced_dim)
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
        if alpha < 1e-10:
          return 0
      return alpha
  
  def __direction__(self,projected_grad,active_constraints_projected_grads,delta1,eps2,dim,reduced_dim):
    if active_constraints_projected_grads is None:
      self.lk = None
      A = None
      direction1 =  -projected_grad
    else:
      b = active_constraints_projected_grads@projected_grad
      A = active_constraints_projected_grads@active_constraints_projected_grads.transpose(0,1)
      self.lk = torch.linalg.solve(A,-b)
      direction1 = -projected_grad - active_constraints_projected_grads.transpose(0,1)@self.lk
    self.grad_norm = torch.linalg.norm(direction1)
    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        l = self.lk.clone()
        l[l>0] = 0
        l *= reduced_dim/dim
        direction2 = -active_constraints_projected_grads.transpose(0,1)@torch.linalg.solve(A,-l)
        return direction2
    else:
      return direction1
  
  def __clear__(self):
    self.first_check = False
    return super().__clear__()
  
  def __run_init__(self, f, con, x0, iteration):
    self.save_values["active"] = torch.zeros(iteration+1)
    self.save_values["grad_norm"] = torch.zeros(iteration+1)  
    return super().__run_init__(f, con, x0, iteration)
  
  def run(self, f, con, x0, iteration, params, save_path, log_interval=-1):
    self.reduced_dim = params["reduced_dim"]
    return super().run(f, con, x0, iteration, params, save_path, log_interval)

  def update_save_values(self, iter, **kwargs):
    self.save_values["active"][iter] = self.active_num
    self.save_values["grad_norm"][iter] = self.grad_norm
    return super().update_save_values(iter, **kwargs)
  
  def get_active_constraints(self,constraints_grads_norm,constraints_values,eps0):
    index = constraints_values > -eps0*constraints_grads_norm  
    return index
  
  def get_projected_gradient_by_matmul(self,Mk,G):
    # Mk:(n,d) Gk(m,n)
    # Mk = None = identity
    if len(G) == 0:
      return None
    
    if Mk is None:
      return G
    else:
      return G@Mk.transpose(0,1) 

  def get_active_constraints_grads(self,eps0):
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    constraints_grads_norm = torch.linalg.norm(constraints_grads,dim = 1)
    active_constraints_index = self.get_active_constraints(constraints_grads_norm=constraints_grads_norm,
                                                           constraints_values=constraints_values,
                                                           eps0=eps0)
    active_constraints_grads = constraints_grads[active_constraints_index]
    self.active_num = len(active_constraints_grads)
    return active_constraints_grads

  def check_norm(self,d,delta1):
    return torch.linalg.norm(d) <= delta1
  
  def check_lambda(self,eps2):
    if self.lk is None:
      return True
    return torch.min(self.lk) > -eps2
   
  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      P = torch.randn(reduced_dim,dim,device = self.device,dtype=self.dtype)/dim
      return P
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
    Gk = self.get_active_constraints_grads(eps0)
    Mk = self.generate_matrix(dim,reduced_dim,mode)
    GkMk = self.get_projected_gradient_by_matmul(Mk,Gk)
    # M_k^\top \nabla f
    projected_grad = self.subspace_first_order_oracle(self.xk,Mk)
    d = self.__direction__(projected_grad,active_constraints_projected_grads=GkMk,delta1=delta1,eps2=eps2,dim=dim,reduced_dim=reduced_dim,r = r)
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
  
  def __direction__(self,projected_grad,active_constraints_projected_grads,delta1,eps2,dim,reduced_dim,r):
    if active_constraints_projected_grads is None:
      self.lk = None
      A = None
      direction1 =  -projected_grad
    else:
      GMMf = active_constraints_projected_grads@projected_grad
      GMMG = active_constraints_projected_grads@active_constraints_projected_grads.transpose(0,1)
      wk = torch.linalg.norm(active_constraints_projected_grads,dim = 1)
      GMMG_inv = torch.linalg.inv(GMMG)
      rk = r/torch.sqrt(GMMG_inv@wk@wk)
      projected_grad_norm = torch.linalg.norm(projected_grad)
      v = rk*wk/projected_grad_norm
      self.lk = -GMMG_inv@GMMf
      lk_bar = -(GMMG_inv@(inverse_xy(v,self.lk)@(GMMf - rk*projected_grad_norm*wk)))
      direction1 = -projected_grad - active_constraints_projected_grads.transpose(0,1)@lk_bar

    self.grad_norm = torch.linalg.norm(direction1)
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