import torch
from torch.autograd.functional import hessian

# Implemented by Pytorch

class constraints:
    def __init__(self,params):
        self.params = params
    
    def set_type(self,dtype):
        # set tensor dtype in params
        for i in range(len(self.params)):
            if isinstance(self.params[i],torch.Tensor):
                self.params[i] = self.params[i].to(dtype)
        return
    
    def get_number_of_constraints(self):
        return

    def __call__(self,x):
        # evaluate function values
        return
    
    def set_device(self,device):
        # set tensor device in params
        for i in range(len(self.params)):
            if isinstance(self.params[i],torch.Tensor):
                self.params[i] = self.params[i].to(device)
        return
 
    def is_feasible(self,x):
        # check feasibility
        # not recommended: time consuming
        with torch.no_grad():
            g_values = self(x)
            return torch.all(g_values<=0)

    def grad(self,x):
        # output: (constraints_num, dim)
        y = x.detach().requires_grad_(True)
        a = self(y)
        output = torch.zeros(a.shape[0],x.shape[0]).to(x.dtype).to(x.device)
        for i in range(a.shape[0]):
            a[i].backward(retain_graph = True)
            output[i] = y.grad
            y.grad = None
        return output
    
    def second_order_oracle(self,x,l):
        # output: \sum_{i} l_i \nabla^2 g_i(x)
        def func(x):
            values = self(x)
            return values@l        
        hess = hessian(func,x)
        return hess
 
class Polytope(constraints):
    # Ax - b
    def __init__(self, params):
        # params: [A,b] 
        super().__init__(params)
    
    def __call__(self,x):
        return self.params[0]@x-self.params[1]
    
    def grad(self, x):
        return self.params[0]
    
    def second_order_oracle(self, x, l):
        return 0
    
    def get_number_of_constraints(self):
        return self.params[0].shape[0]

class NonNegative(constraints):
    # x >= 0
    def __init__(self, params):
        # params = [dim]
        super().__init__(params)

    def grad(self,x):
        return -torch.eye(x.shape[0],dtype = x.dtype,device=x.device)

    def __call__(self,x):
        return -x

    def second_order_oracle(self, x, l):
        return 0
    
    def get_number_of_constraints(self):
        return self.params[0]

class Quadratic(constraints):
    # 1/2*x Q_i x + b_ix + c_i 
    # Q: symmetric
    def grad(self,x):
        return self.params[0]@x + self.params[1]

    def __call__(self,x):
        return 1/2*self.params[0]@x@x + self.params[1]@x + self.params[2]
    
    def second_order_oracle(self, x, l):
        k = l.view(-1, 1, 1)
        output = torch.sum(k * self.params[0], dim=0)
        return output
    
    def get_number_of_constraints(self):
        return self.params[0].shape[0]
      
class Fused_Lasso(constraints):
    def function(self,x):
        return torch.sum(torch.abs(x))

    def function2(self,x):
        return torch.sum(torch.abs(x[1:]-x[0:-1]))
    
    def __call__(self, x):
        with torch.no_grad():
            return torch.Tensor([self.function(x)-self.params[0],self.function2(x)-self.params[1]]).to(x.device)
 
    def grad(self,x):
        y = x.detach()
        y.requires_grad_(True)
        self.function(y).backward()
        g1 = y.grad
        y.grad = None
        self.function2(y).backward()
        g2 = y.grad
        y.grad = None
        return torch.cat([g1.reshape(1,-1),g2.reshape(1,-1)],dim = 0).to(x.device)

    def get_number_of_constraints(self):
        return 2
    
class Ball(constraints):
    def __call__(self, x):
        return ( torch.linalg.norm(x,ord = self.params[1])**self.params[1]-self.params[0]).reshape(1)
    def grad(self,x):
        y = x.detach().clone()
        y.requires_grad_(True)
        self(y).backward()
        return y.grad.reshape(1,-1)
    
    def get_number_of_constraints(self):
        return 1
    
class Huber(constraints):
    def __call__(self,x):
        index = x < self.params[0]
        a = torch.sum(1/2*x[index]**2)
        b = torch.sum( self.params[0]*(torch.abs(x[torch.logical_not(index)] - 0.5*self.params[1])))
        return (a +b - self.params[1]).reshape(1)

    def grad(self,x):
        y = x.detach().clone()
        y.requires_grad_(True)
        self.Value(y).backward()
        return y.grad.reshape(1,-1)

    def get_number_of_constraints(self):
        return 1

# class LogSumExp(constraints):
#     def __init__(self, params, con,eps=0):
#         super().__init__(params, eps)
#         self.constraints = con

#     def Name(self):
#         return "LogSumExp"
    
#     def LogSumExp_value(self,x,t = 1):
#         return torch.log(torch.sum(torch.exp(t*x)))/t
  
#     def LogSumExpGrad(self,x,t = 1):
#         return torch.exp(t*x)/torch.sum(torch.exp(t*x))
    
#     def D(self,x):
#         y = x.detach().clone()
#         y.requires_grad_(True)
#         self.Value(y).backward()
#         return y.grad.reshape(1,-1)

#     def Value(self,x):
#         return self.LogSumExp_value(self.constraints.Value(x),t = torch.log(torch.tensor(1 + x.shape[0]))).reshape(1) - 1 -self.error
    

    
# class Fairness(constraints):
#     def __init__(self, params, eps=0):
#         super().__init__(params, eps)
#         self.z_mean = torch.mean(self.params[1])
    
#     def SetDevice(self, device):
#         super().SetDevice(device)
#         self.z_mean = self.z_mean.to(device)
    
#     def SetDtype(self, dtype):
#         super().SetDtype(dtype)
#         self.z_mean = self.z_mean.to(dtype)
    
#     def Name(self):
#         return "Fairness"

#     def d_theta(self,x):
#         # 各データに対応する d_\thetaを出力する
#         return self.params[0]@x[:self.params[3].to(torch.int64)]

#     def Value(self,x):
#         v1 = torch.mean( (self.params[1]- self.z_mean)*self.d_theta(x) ) -self.params[2]
#         v2 = -torch.mean( (self.params[1]- self.z_mean)*self.d_theta(x) ) -self.params[2]
#         return torch.tensor([v1,v2],device=x.device,dtype=x.dtype)

#     def D(self,x):
#         y1 = x.detach().clone()
#         y1.requires_grad_(True)
#         v1 = torch.mean( (self.params[1]- self.z_mean)*self.d_theta(y1) ) -self.params[2]
#         v1.backward()
#         return torch.concat([y1.grad.reshape(1,-1),-y1.grad.reshape(1,-1)],dim = 0)
