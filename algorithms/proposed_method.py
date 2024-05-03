from algorithms.constrained_descent_method import constrained_optimization_solver
from algorithms.zeroth_method import zeroth_solver
from utils.calculate import inverse_xy,jax_randn,subspace_line_search
from utils.logger import logger
from utils.select import get_step_scheduler_func
import numpy as np
from environments import DIRECTIONALDERIVATIVE,FINITEDIFFERENCE
import jax.numpy as jnp
from jax import grad,jit


class RSGLC(constrained_optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    self.lk = None
    self.first_check = False
    self.active_num = 0
    self.grad_norm = 0
    self.reduced_dim = 0
    super().__init__(dtype)
    self.params_key = ["eps0",
                  "delta1",
                  "eps2",
                  "dim",
                  "reduced_dim",
                  "alpha1",
                  "alpha2",
                  "delta",
                  "beta",
                  "mode",
                  "backward"]
  

  def __iter_per__(self):
    eps0 = self.params["eps0"]
    delta1 = self.params["delta1"]
    eps2 = self.params["eps2"]
    dim = self.params["dim"]
    mode = self.params["mode"]
    alpha1 = self.params["alpha1"]
    alpha2 = self.params["alpha2"]
    beta = self.params["beta"]
    delta = self.params["delta"]
    Gk = self.get_active_constraints_grads(eps0)
    # while self.reduced_dim-self.active_num < 5:
    #   self.reduced_dim += 10
    #   if self.reduced_dim > dim:
    #     self.reduced_dim = dim
    #     break
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
      Md = Mk.T@d
    if self.first_check:
      alpha =self.__step_size__(Md,alpha2,beta,delta)
    else:
      alpha = self.__step_size__(Md,alpha1,beta,delta)
    
    self.__update__(alpha*Md)
    return
    
  def __step_size__(self,direction,alpha,beta,delta):
    while not self.con.is_feasible(self.xk + alpha*direction):
      alpha *= beta
      if alpha < 1e-30:
        return 0
    while self.f(self.xk + alpha*direction) -self.f(self.xk) >0:
      alpha *= beta
    if self.f(self.xk + alpha*direction) - self.f(self.xk) < -delta:
      return alpha
    else:
      return 0
  
  def __direction__(self,projected_grad,active_constraints_projected_grads,delta1,eps2,dim,reduced_dim):
    if active_constraints_projected_grads is None:
      self.lk = None
      A = None
      direction1 =  -projected_grad
    else:
      b = active_constraints_projected_grads@projected_grad
      A = active_constraints_projected_grads@active_constraints_projected_grads.T
      self.lk = jnp.linalg.solve(A,-b)
      direction1 = -projected_grad - active_constraints_projected_grads.T@self.lk
    self.grad_norm = jnp.linalg.norm(direction1)
    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        l = np.array(self.lk)
        l[l>0] = 0
        l = reduced_dim/dim*jnp.array(l)
        direction2 = -active_constraints_projected_grads.T@jnp.linalg.solve(A,-l)
        return direction2
    else:
      return direction1
  
  def __clear__(self):
    self.first_check = False
    return super().__clear__()
  
  def __run_init__(self, f, con, x0, iteration,params):
    self.save_values["active"] = np.zeros(iteration+1)
    self.save_values["grad_norm"] = np.zeros(iteration+1)  
    return super().__run_init__(f, con, x0, iteration,params)
  
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
      return G@Mk.T 

  def get_active_constraints_grads(self,eps0):
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    constraints_grads_norm = jnp.linalg.norm(constraints_grads,axis = 1)
    active_constraints_index = self.get_active_constraints(constraints_grads_norm=constraints_grads_norm,
                                                           constraints_values=constraints_values,
                                                           eps0=eps0)
    active_constraints_grads = constraints_grads[active_constraints_index]
    self.active_num = len(active_constraints_grads)
    return active_constraints_grads

  def check_norm(self,d,delta1):
    return jnp.linalg.norm(d) <= delta1
  
  def check_lambda(self,eps2):
    if self.lk is None:
      return True
    return jnp.min(self.lk) > -eps2
   
  def generate_matrix(self,dim,reduced_dim,mode):
    # (dim,reduced_dim)の行列を生成
    if mode == "random":
      P = jax_randn(reduced_dim,dim,dtype=self.dtype)/dim
      return P
    elif mode == "identity":
      return None
    else:
      raise ValueError("No matrix mode")

class RSGLC_norandom(constrained_optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    self.lk = None
    self.first_check = False
    self.active_num = 0
    self.grad_norm = 0
    super().__init__(dtype)
    self.params_key = ["eps0",
                  "delta1",
                  "eps2",
                  "dim",
                  "alpha1",
                  "alpha2",
                  "delta",
                  "beta",
                  "backward"]
  

  def __iter_per__(self):
    eps0 = self.params["eps0"]
    delta1 = self.params["delta1"]
    eps2 = self.params["eps2"]
    dim = self.params["dim"]
    alpha1 = self.params["alpha1"]
    alpha2 = self.params["alpha2"]
    beta = self.params["beta"]
    delta = self.params["delta"]
    Gk = self.get_active_constraints_grads(eps0)
    Gk = self.get_projected_gradient_by_matmul(None,Gk)
    # M_k^\top \nabla f
    grad = self.__first_order_oracle__(self.xk)
    d = self.__direction__(grad,active_constraints_grads=Gk,delta1=delta1,eps2=eps2,dim=dim)
    if self.first_check:
      alpha =self.__step_size__(d,alpha2,beta,delta)
    else:
      alpha = self.__step_size__(d,alpha1,beta,delta)
    self.__update__(alpha*d)
    return
    
  def __step_size__(self,direction,alpha,beta,delta):
    while not self.con.is_feasible(self.xk + alpha*direction):
      alpha *= beta
      if alpha < 1e-30:
        return 0
    while self.f(self.xk + alpha*direction) -self.f(self.xk) >0:
      alpha *= beta

    if self.f(self.xk + alpha*direction) - self.f(self.xk) < -delta:
      return alpha
    else:
      return 0
  
  def __direction__(self,grad,active_constraints_grads,delta1,eps2,dim):
    if active_constraints_grads is None:
      self.lk = None
      A = None
      direction1 =  -grad
    else:
      b = active_constraints_grads@grad
      A = active_constraints_grads@active_constraints_grads.T
      self.lk = jnp.linalg.solve(A,-b)
      direction1 = -grad - active_constraints_grads.T@self.lk
    self.grad_norm = jnp.linalg.norm(direction1)
    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        l = np.array(self.lk)
        l[l>0] = 0
        l = jnp.array(l)
        direction2 = -active_constraints_grads.T@jnp.linalg.solve(A,-l)
        return direction2
    else:
      return direction1
  
  def __clear__(self):
    self.first_check = False
    return super().__clear__()
  
  def __run_init__(self, f, con, x0, iteration,params):
    self.save_values["active"] = np.zeros(iteration+1)
    self.save_values["grad_norm"] = np.zeros(iteration+1)  
    return super().__run_init__(f, con, x0, iteration,params)
  
  def run(self, f, con, x0, iteration, params, save_path, log_interval=-1):
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
      return G@Mk.T 

  def get_active_constraints_grads(self,eps0):
    constraints_values = self.evaluate_constraints_values(self.xk)
    constraints_grads = self.evaluate_constraints_grads(self.xk)
    constraints_grads_norm = jnp.linalg.norm(constraints_grads,axis = 1)
    active_constraints_index = self.get_active_constraints(constraints_grads_norm=constraints_grads_norm,
                                                           constraints_values=constraints_values,
                                                           eps0=eps0)
    active_constraints_grads = constraints_grads[active_constraints_index]
    self.active_num = len(active_constraints_grads)
    return active_constraints_grads

  def check_norm(self,d,delta1):
    return jnp.linalg.norm(d) <= delta1
  
  def check_lambda(self,eps2):
    if self.lk is None:
      return True
    return jnp.min(self.lk) > -eps2
   
  

class RSGNC(RSGLC):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
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
      
  def __iter_per__(self):
    # 後でx.gradについて確認
    eps0 = self.params["eps0"]
    delta1 = self.params["delta1"]
    eps2 = self.params["eps2"]
    dim = self.params["dim"]
    reduced_dim = self.params["reduced_dim"]
    mode = self.params["mode"]
    alpha1 = self.params["alpha1"]
    alpha2 = self.params["alpha2"]
    beta = self.params["beta"]
    r = self.params["r"]
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
      Md = Mk.T@d
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
      GMMG = active_constraints_projected_grads@active_constraints_projected_grads.T
      wk = jnp.linalg.norm(active_constraints_projected_grads,axis = 1)
      GMMG_inv = jnp.linalg.inv(GMMG)
      rk = r/jnp.sqrt(GMMG_inv@wk@wk)
      projected_grad_norm = jnp.linalg.norm(projected_grad)
      v = rk*wk/projected_grad_norm
      self.lk = -GMMG_inv@GMMf
      lk_bar = -(GMMG_inv@(inverse_xy(v,self.lk)@(GMMf - rk*projected_grad_norm*wk)))
      direction1 = -projected_grad - active_constraints_projected_grads.T@lk_bar

    self.grad_norm = jnp.linalg.norm(direction1)
    if self.check_norm(direction1,delta1):
      self.first_check = True
      if self.check_lambda(eps2):
        self.finish = True
        return None
      else:
        if -jnp.sum(self.lk) >= eps2/2:
          s = jnp.ones(self.lk.shape[0],dtype= self.dtype)
        else:
          s = np.ones(self.lk.shape[0], dtype= self.dtype)
          negative_sum = jnp.sum(self.lk[self.lk<0])
          positive_sum = jnp.sum(self.lk[self.lk>0])
          s[self.lk>0] = -negative_sum/positive_sum/2
          s = jnp.array(s)
        direction2 = -eps2*reduced_dim/dim*active_constraints_projected_grads.T@(GMMG_inv@s)
        return direction2
    else:
      return direction1

class RSRGF(zeroth_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.step_scheduler_func = None
    self.params_key = [
      "lr",
      "sample_size",
      "reduced_dim",
      "mu",
      "schedule",
      "central",
      "projection"
    ]
  
  def __run_init__(self, f, x0, iteration, params):
    super().__run_init__(f, x0, iteration, params)
    self.step_scheduler_func = get_step_scheduler_func(params["schedule"])

  def __direction__(self,loss):
    reduced_dim = self.params["reduced_dim"]
    sample_size = self.params["sample_size"]
    mu = self.params["mu"]
    dim = self.xk.shape[0]
    P = jax_randn(dim,reduced_dim,dtype = self.dtype)/(dim**0.5)
    subspace_dir = None
    if self.params["projection"]:
      U = jax_randn(sample_size,reduced_dim+1,dtype = self.dtype)/(sample_size**0.5)
      U = U.at[:, 0].set(1)
    else:
      U = jax_randn(sample_size,reduced_dim,dtype = self.dtype)/(sample_size**0.5)
    
    if self.params["central"]:
      if self.params["projection"]:
        for i in range(sample_size):
          m = mu*P@U[i,1:]
          g1 = self.func(self.xk + m,u= mu*U[i])
          g2 = self.func(self.xk - m,u= mu*U[i])
          if subspace_dir is None:
            subspace_dir = (g1 - g2)/(2*mu) * U[i,1:]
          else:
            subspace_dir += (g1 - g2)/(2*mu) * U[i,1:]
      else:
        for i in range(sample_size):
          m = mu*P@U[i]
          g1 = self.func(self.xk + m)
          g2 = self.func(self.xk - m)
          if subspace_dir is None:
            subspace_dir = (g1 - g2)/(2*mu) * U[i]
          else:
            subspace_dir += (g1 - g2)/(2*mu) * U[i]
    else:
      for i in range(sample_size):
        g1 = self.func(self.xk + m)
        if subspace_dir is None:
          subspace_dir = (g1 - loss.item())/mu * U[i]
        else:
          subspace_dir += (g1 - loss.item())/mu * U[i]
    return - P@subspace_dir

  def __step_size__(self, iter):
    return self.step_scheduler_func(iter)*self.params["lr"]
    
  def __iter_per__(self, iter):
    loss = self.save_values["func_values"][iter]
    dk = self.__direction__(loss)
    lr = self.__step__(iter)
    self.__update__(lr*dk)