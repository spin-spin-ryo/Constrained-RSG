from jax import random
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
key = random.PRNGKey(0)
DATAPATH = "./data"
RESULTPATH = "./results"
DTYPE = jnp.float64

# 目的関数一覧
QUADRATIC = "Quadratic"
SPARSEQUADRATIC = "SparseQuadratic"
MATRIXFACTORIZATION = "MatrixFactorization"
MATRIXFACTORIZATION_COMPLETION = "MatrixFactorization_Completions"
LEASTSQUARE = "LeastSquare"
MLPNET = "MLPNET"
CNN = "CNN"
SOFTMAX = "Softmax"
LOGISTIC = "Logistic"
SPARSEGAUSSIANPROCESS = "SparseGaussianProcess"
REGULARIZED = "Regularized"

# 制約一覧
POLYTOPE = "Polytope"
NONNEGATIVE = "NonNegative"
FUSEDLASSO = "FusedLasso"
BALL = "Ball"
HUBER = "Huber"
NOCONSTRAINTS = "NoConstraint"

# アルゴリズム手法一覧
GRADIENT_DESCENT = "GD"
SUBSPACE_GRADIENT_DESCENT = "SGD"
ACCELERATED_GRADIENT_DESCENT = "AGD"
NEWTON = "Newton"
SUBSPACE_NEWTON = "RSNewton"
LIMITED_MEMORY_NEWTON = "LMN"
LIMITED_MEMORY_BFGS = "LM_BFGS"
BFGS_QUASI_NEWTON = "bfgs"
RANDOM_BFGS = "Randomized_bfgs"
SUBSPACE_REGULARIZED_NEWTON = "RSRNM" 
SUBSPACE_QUASI_NEWTON = "Proposed"
PROXIMAL_GRADIENT_DESCENT = "PGD"
ACCELERATED_PROXIMAL_GRADIENT_DESCENT = "APGD"
MARUMO_AGD = "AGD(2022)"
GRADIENT_PROJECTION = "GPD"
DYNAMIC_BARRIER = "Dynamic"
PRIMALDUAL = "PrimalDual"
RSG_LC = "proposed_linear"
G_LC = "proposed_linear_norandom"
RSG_NC = "proposed_nonlinear"
G_NC = "proposed_nonlinear_norandom"
LINESEARCH_RSGLC = "linesearch_proposed_linear"
RGF = "RGF"
RSRGF = "proposed_zeroth"

# ディレクトリ名の形式 
# results/{objectives}/{param}@{value}~{param}@{value}~..../{constraints}/{param}@{value}~{param}@{value}~..../{solver_name}/{param}@{value}~{param}@{value}

DISTINCT_PARAM_VALUE = "@"
DISTINCT_PARAMS = "~"

# 勾配計算の仕方
DIRECTIONALDERIVATIVE = "DD"
FINITEDIFFERENCE = "FD"

# スケッチ行列の決め方
RANDOM = "random"
LEESELECTION = "Lee"

objective_properties_key ={
    QUADRATIC:["dim","convex","data_name"],
    SPARSEQUADRATIC:["dim","data_name"],
    MATRIXFACTORIZATION:["data_name","rank"],
    MATRIXFACTORIZATION_COMPLETION:["data_name","rank"],
    LEASTSQUARE:["data_name","data_size","dim"],
    MLPNET: ["data_name","layers_size","activation","criterion"],
    CNN: ["data_name","layers_size","activation","criterion"],
    SOFTMAX:["data_name"],
    LOGISTIC:["data_name"],
    SPARSEGAUSSIANPROCESS:["data_name","reduced_data_size","kernel_mode"],
    REGULARIZED: ["coeff","ord","Fused"]
}

constraints_properties_key = {
    POLYTOPE:["data_name","dim","constraints_num"],
    NONNEGATIVE:["dim"],
    QUADRATIC:["data_name","dim","constraints_num"],
    FUSEDLASSO: ["threshold1","threshold2"],
    BALL:["ord","threshold"],
    HUBER:["delta","threshold"]
}

algorithm_parameters_key = {
  GRADIENT_DESCENT:["lr","eps","backward","linesearch"],
  SUBSPACE_GRADIENT_DESCENT:["lr","reduced_dim","dim","mode","eps","backward","linesearch"],
  ACCELERATED_GRADIENT_DESCENT:["lr","eps","backward","restart"],
  BFGS_QUASI_NEWTON:["alpha","beta","backward","eps"],
  LIMITED_MEMORY_BFGS:["alpha","beta","backward","eps","memory_size"],
  PROXIMAL_GRADIENT_DESCENT:["eps","beta","backward","alpha"],
  ACCELERATED_PROXIMAL_GRADIENT_DESCENT:["restart","beta","eps","backward","alpha"],
  NEWTON:["alpha","beta","eps","backward"],
  SUBSPACE_NEWTON:["dim","reduced_dim","mode","backward","alpha","beta","eps"],
  LIMITED_MEMORY_NEWTON:["reduced_dim","threshold_eigenvalue","alpha","beta","backward","mode","eps"],
  RANDOM_BFGS:["reduced_dim","dim","backward","eps"],
  SUBSPACE_REGULARIZED_NEWTON:["reduced_dim","gamma","c1","c2","alpha","beta","eps","backward"],
  GRADIENT_PROJECTION:["eps","delta","lr","alpha","beta","backward"],
  DYNAMIC_BARRIER:["lr","alpha","beta","barrier_func_type","sub_problem_eps","inner_iteration","backward"],
  PRIMALDUAL:["mu","eps","eps_feas","beta","alpha","backward"],
  RSG_LC: ["eps0","delta1","eps2","dim","reduced_dim","alpha","beta","mode","backward"],
  G_LC:["eps0","delta1","eps2","dim","alpha","beta","backward"],
  RSG_NC: ["eps0","delta1","eps2","dim","reduced_dim","alpha","beta","mode","r","backward"],
  G_NC: ["eps0","delta1","eps2","dim","alpha","beta","r","backward"]
}


