from algorithms.descent_method import optimization_solver
from jax.numpy import float64
import jax.numpy as jnp
import numpy as np
import time
from utils.logger import logger
from utils.calculate import jax_randn
from utils.select import get_step_scheduler_func

class zeroth_solver(optimization_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.backward_mode = False
  
  def __run_init__(self, f, x0, iteration, params):
    self.f = f
    self.xk = x0.copy()
    self.save_values["func_values"] = np.zeros(iteration+1)
    self.save_values["time"] = np.zeros(iteration+1)
    self.finish = False
    self.params = params
    self.check_count = 0
    self.save_values["func_values"][0] = self.f(self.xk)
  
  def __iter_per__(self,iter):
    return
  
  def __step_size__(self,iter):
    return 
  
  def run(self,f,x0,iteration,params,save_path,log_interval = -1):
    self.__check_params__(params)
    self.__run_init__(f,x0,iteration,params)
    start_time = time.time()
    for i in range(iteration):
      self.__clear__()
      self.__iter_per__(i)
      T = time.time() - start_time
      F = self.f(self.xk)
      self.update_save_values(i+1,time = T,func_values = F)
      if (i+1)%log_interval == 0 and log_interval != -1:
        logger.info(f'{i+1}: {self.save_values["func_values"][i+1]}')
        self.save_results(save_path)
    return
    

class random_gradient_free(zeroth_solver):
  def __init__(self, dtype=jnp.float64) -> None:
    super().__init__(dtype)
    self.step_scheduler_func = None
    self.params_key = [
      "mu",
      "sample_size",
      "lr",
      "schedule",
      "central"
    ]
  
  def __run_init__(self, f, x0, iteration, params):
    super().__run_init__(f, x0, iteration, params)
    self.step_scheduler_func = get_step_scheduler_func(params["schedule"])

  def __direction__(self,loss):
    mu = self.params["mu"]
    sample_size = self.params["sample_size"]
    dim = self.xk.shape[0]
    dir = None
    P = jax_randn(sample_size,dim,dtype = self.dtype)/(sample_size**(0.5))
    for i in range(sample_size):
      if self.params["central"]:
        f1 = self.func(self.xk + mu*P[i])
        f2 = self.func(self.xk - mu*P[i])
        if dir is None:
          dir = (f1 - f2)/(2*mu) *P[i]
        else:
          dir += (f1 - f2)/(2*mu) *P[i]
      else:
        f1 = self.func(self.xk + mu*P[i])
        if dir is None:
          dir = (f1 - loss)/mu * P[i] 
        else:
          dir += (f1 - loss)/mu * P[i]
    return - dir 
  
  def __step_size__(self, iter):
    return self.step_scheduler_func(iter)*self.params["lr"]
    
  
  def __iter_per__(self, iter):
    loss = self.save_values["func_values"][iter]
    dk = self.__direction__(loss)
    lr = self.__step__(iter)
    self.__update__(lr*dk)
        
  

  
