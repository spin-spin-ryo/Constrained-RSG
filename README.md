# RSG-LC and RSG-NC

## How to run numerical experiments
1. Select optimization problem and an algorithm and set the parameters in `make_config.py`.
2. Make config.json file using command `python make_config.py`.
3. Run numerical experiments using command `python main.py path/to/config.json`.
4. Check the results in `results/problem_name/problem_parameters/constraints_name/constraints_parameters/algorithm_name/algorithm_parameters` directory.
5. You can compare results using `python result_show.py`. with GUI interface.

## problems

### QUADRATIC
$$\min_{x\in \mathbb{R}^n} \frac{1}{2}x^\top A x + b^\top x$$
dim: n, if convex is True, then A become a semi definite matrix. data_name: only "random".

### SPARSEQUADRATIC
not used

### MATRIXFACTORIZATION
$$\min_{U,V} \|UV - X\|_F^2$$
data_name: only "movie", rank: the number of row of $U$, and column of $V$.

### MATRIXFACTORIZATION_COMPLETION
$$\min_{U,V} \|\mathcal{P}_{\Omega}(UV) - \mathcal{P}_{\Omega}(X)\|_F^2$$
data_name: only "movie", rank: the number of row of $U$, and column of $V$.

### LEASTSQUARE
not used

### MLPNET
linear neural network:
$$\min_{w} \sum_{i=1}^m \mathcal{L}(\mathcal{W}(w,x_i),y_i)$$
$(x_i,y_i)$:dataset, layers_size: [(in_features,out_feafures,use bias or not),], activation: activation function name (see `utils/select.py`), criterion: type of loss function (only 'CrossEntropy')

### CNN
convolutional neural network:
$$\min_{w} \sum_{i=1}^m \mathcal{L}(\mathcal{W}(w,x_i),y_i)$$
$(x_i,y_i)$:dataset,
layers_size: [(input_channels,output_channelskernel_size,bias_flag)],
activation: activation function name (see `utils/select.py`),
criterion: type of loss function (only 'CrossEntropy')

### SOFTMAX
minimizing softmax loss function.</br>
data_name: "Scotus" or "news20"

### LOGISTIC
minimizing logistic loss function</br>
data_name:"rcv1" or "news20" or "random".

### REGULARIZED
set `problem_name = REGULARIZED + other_problem_name`.
minimizing regularized function 
$$\min_x f(x) + \lambda \|x\|_p^p$$
coeff: $\lambda$,
ord: $p$,
Fused: only False

## constraints

### POLYTOPE
$$\{ x| Ax-b \le 0\}$$
data_name:only "random",
dim: the dimension of $x$,
constraints_num: the dimension of $b$

### NONNEGATIVE
$$\{x | x_i \ge 0\}$$
dim: the dimension of $x$

### QUADRATIC
$$\{x| \frac{1}{2}x^\top A_i x + b_i^\top x + c_i \le 0, i = 1,...,m\}$$
data_name: only "random",
dim: the dimension of $x$,
constraints_num: m

### FUSEDLASSO
$$\{x| \|x\|_1 \le s_1, \sum_{i} |x_{i+1} - x_i|\le s_2\}$$
threshold1: $s_1$,
threshold2: $s_2$

### BALL
$$\{x| \|x\|_p^p \le s\}$$
ord: $p$,
threshold: $s$

## algorithms
#### backward parameters
 True: use automatic differentiation (very fast, but memory leak sometimes happens (CNN))
  DD: use directional derivative with automatic differentiation (efficiency depends on the dimension, no error)
  FD: use finite difference (efficiency depends on the dimension, error exists)

### GradientProjectionMethod[https://epubs.siam.org/doi/abs/10.1137/0108011?journalCode=smjmap.1]
eps: the parameter of active set,
delta: criteria of gradient norm, 
alpha: step size,
beta: the parameter of linesearch

### DynamicBarrierGD(Dynamic Barrier Gradient descent[https://proceedings.neurips.cc/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html])
lr: step size,
alpha: the parameter of barrier function (see original paper),
beta: the parameter of barrier function (see original paper),
barrier_func_type: BARRIERTYPE1 or BARRIERTYPE2,
sub_problem_eps: when the number of constraints is more than one, this method solve subproblems. stop criteria.
inner_iteration: iteration of solving subproblem,
    
### PrimalDualInteriorPointMethod(https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
see text book.
mu(>1),eps,eps_feas,
beta: the parameter of line search,
alpha: the parameter of line search

### RSGLC(Proposed method for linear constraint)
eps0: the parameter of active constraints,
delta1: the parameter of gradient norm,
eps2: the parameter of Lagrange multiplier,
dim: the dimension of optimization problem,
reduced_dim: the size of random matrix,
alpha1: initial step size of first direction,
alpha2: initial step size of second direction,
beta: the parameter of line search,
mode: the type of random matrix(only "random"),

### RSGNC(Proposed method for nonlinear constraint)
eps0: the parameter of active constraints,
delta1: the parameter of gradient norm,
eps2: the parameter of Lagrange multiplier,
dim: the dimension of optimization problem,
reduced_dim: the size of random matrix,
alpha1: initial step size of first direction,
alpha2: initial step size of second direction,
beta: the parameter of line search,
r: the parameter of first direction,
mode: the type of random matrix(only "random"),

