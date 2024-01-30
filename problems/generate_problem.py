from problems.objectives import *
from problems.constraints import *
from environments import *
import pickle
import torch.nn.functional as F
import torch.nn as nn
from utils.calculate import generate_symmetric,generate_semidefinite,nonnegative_projection,L1projection,BallProjection,BoxProjection
from utils.select import get_activation,get_criterion
import os

objective_properties_key ={
    QUADRATIC:["dim","convex","data_name"],
    SPARSEQUADRATIC:["dim","data_name"],
    MATRIXFACTORIZATION:["data_name","rank"],
    MATRIXFACTORIZATION_COMPLETION:["data_name","rank"],
    LEASTSQUARE:["data_name","data_size","dim"],
    MLPNET: ["data_name","layers_size","activation","criterion"],
    CNN: ["data_name","layers_size","activation","criterion"]
}

constraints_properties_key = {
    POLYTOPE:["data_name","dim","constraints_num"],
    NONNEGATIVE:["dim"],
    QUADRATIC:["data_name","dim","constraints_num"],
    FUSEDLASSO: ["threshold1","threshold2"],
    BALL:["ord","threshold"],
    HUBER:["delta","threshold"]
}

def generate_objective(function_name,function_properties):
    if function_name == QUADRATIC:
        f = generate_quadratic(function_properties)
    elif function_name == SPARSEQUADRATIC:
        f = generate_sparse_quadratic(function_properties)
    elif function_name == MATRIXFACTORIZATION:
        f = generate_matrix_factorization(function_properties)
    elif function_name == MATRIXFACTORIZATION_COMPLETION:
        f = generate_matrix_factorization_completion(function_properties)
    elif function_name == LEASTSQUARE:
        f = generate_least_square(function_properties)
    elif function_name == MLPNET:
        f = generate_mlpnet(function_properties)
    elif function_name == CNN:
        f = generate_cnn(function_properties)
    else:
        raise ValueError(f"{function_name} is not implemented.")
    return f

def generate_constraints(constraints_name, constraints_properties):
    prox = None
    if constraints_name == POLYTOPE:
        constraints = generate_polytope(constraints_properties)
    elif constraints_name == NONNEGATIVE:
        prox = nonnegative_projection
        constraints = generate_nonnegative(constraints_properties)
    elif constraints_name == QUADRATIC:
        constraints = generate_quadratic_constraints(constraints_properties)
    elif constraints_name == FUSEDLASSO:
        constraints = generate_fusedlasso(constraints_properties)
    elif constraints_name == BALL:
        ord = constraints_properties["ord"]
        threshold = constraints_properties["threshold"]
        if ord == 1:
            def prox(x,t):
                return L1projection(x,radius=threshold)
        elif ord == 2:
            def prox(x,t):
                return BallProjection(x,radius=threshold**(0.5))
        else:
            raise ValueError("No proximal function")
        constraints = generate_ball(constraints_properties)
    elif constraints_name == HUBER:
        constraints = generate_huber(constraints_properties)
    elif constraints_name == NOCONSTRAINTS:
        constraints = None
    else:
        raise ValueError(f"{constraints_name} is not implemented.")
    return constraints, prox

def generate_initial_points(func,function_name,constraints_name,function_properties):
    dim = func.get_dimension()
    # 非負制約の時のみすべて1
    if constraints_name == NONNEGATIVE:
        x0 = torch.ones(dim)
        return x0
    
    if function_name == MLPNET:
        if function_properties["data_name"] == "mnist":
            # dim:669706
            x0 = torch.load(os.path.join(DATAPATH,"mnist","mlpnet","init_param.pth"))
            return x0
    
    if function_name == CNN:
        if function_properties["data_name"] == "mnist":
            # dim:33738
            x0 = torch.load(os.path.join(DATAPATH,"mnist","cnn","init_param.pth"))
            return x0
    x0 = torch.zeros(dim)
    return x0

def generate_quadratic(properties):
    dim = int(properties["dim"])
    convex = properties["convex"]
    data_name = properties["data_name"]
    if data_name == "random":
        if convex:
            # rankは適当
            rank = dim//2
            Quadratic_data_path = os.path.join(DATAPATH,"quadratic","convex")
            filename_Q = f"Q_{dim}_{rank}.pth"
            filename_b = f"b_{dim}_{rank}.pth"
            if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
                Q = torch.load(os.path.join(Quadratic_data_path,filename_Q))
                b = torch.load(os.path.join(Quadratic_data_path,filename_b))
                c = 0
            else:
                Q = generate_semidefinite(dim=dim,rank=rank,device = DEVICE)
                b = torch.randn(dim,device = DEVICE)
                c = 0
                os.makedirs(Quadratic_data_path,exist_ok=True)
                torch.save(Q,os.path.join(Quadratic_data_path,filename_Q))
                torch.save(b,os.path.join(Quadratic_data_path,filename_b))
        else:   
            Quadratic_data_path = os.path.join(DATAPATH,"quadratic","nonconvex")
            filename_Q = f"Q_{dim}.pth"
            filename_b = f"b_{dim}.pth"
            if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
                Q = torch.load(os.path.join(Quadratic_data_path,filename_Q))
                b = torch.load(os.path.join(Quadratic_data_path,filename_b))
                c = 0
            else:
                Q = generate_symmetric(dim=dim,device = DEVICE)
                b = torch.randn(dim,device = DEVICE)
                c = 0
                os.makedirs(Quadratic_data_path,exist_ok=True)
                torch.save(Q,os.path.join(Quadratic_data_path,filename_Q))
                torch.save(b,os.path.join(Quadratic_data_path,filename_b))                    
    params = [Q,b,c]
    f = QuadraticFunction(params=params)
    return f

def generate_sparse_quadratic(properties):
    dim = int(properties["dim"])
    data_name = properties["data_name"]
    if data_name == "random":
        Quadratic_data_path = os.path.join(DATAPATH,"sparse_quadratic")
        filename_Q = f"Q_{dim}.pth"
        filename_b = f"b_{dim}.pth"
        if os.path.exists(os.path.join(Quadratic_data_path,filename_Q)):
            Q = torch.load(os.path.join(Quadratic_data_path,filename_Q))
            b = torch.load(os.path.join(Quadratic_data_path,filename_b))
            c = 0
        else:
            Q = torch.randn(dim,device = DEVICE)
            b = torch.randn(dim,device = DEVICE)
            c = 0
            os.makedirs(Quadratic_data_path,exist_ok=True)
            torch.save(Q,os.path.join(Quadratic_data_path,filename_Q))
            torch.save(b,os.path.join(Quadratic_data_path,filename_b))    
    params = [Q,b,c]
    f = SparseQuadraticFunction(params=params)
    return f

def generate_matrix_factorization(properties):
    data_name = properties["data_name"]
    rank = int(properties["rank"])
    if data_name == "movie":
        with open(os.path.join(DATAPATH,"movie","movie_100k.pth"),"rb") as data:
            U = pickle.load(data)
    
    params = [U,rank]
    f = MatrixFactorization_2(params=params)
    return f

def generate_matrix_factorization_completion(properties):
    data_name = properties["data_name"]
    rank = int(properties["rank"])
    if data_name == "movie":
        with open(os.path.join(DATAPATH,"movie","movie_train_100k.pth"),"rb") as data:
            U = pickle.load(data)
    
    params = [U,rank]
    f = MatrixFactorization_Completion(params=params)
    return f

def generate_least_square(properties):
    data_name = properties["data_name"]
    data_size = int(properties("data_size"))
    dim = int(properties("dim"))
    if data_name == "random":
        data_path = os.path.join(DATAPATH,"least_square")
        filename_A = f"A_{dim}_{data_size}.pth"
        filename_b = f"b_{dim}_{data_size}.pth"
        if os.path.exists(os.path.join(data_path,filename_A)):
            A = torch.load(os.path.join(data_path,filename_A))
            b = torch.load(os.path.join(data_path,filename_b))
        else:
            A = torch.randn(data_size,dim)
            b = torch.randn(data_size)
            os.makedirs(data_path,exist_ok=True)
            torch.save(A,os.path.join(data_path,filename_A))
            torch.save(b,os.path.join(data_path,filename_b))
    
    params = [A,b]
    f = LeastSquare(params)
    return f

def generate_mlpnet(properties):
    data_name = properties["data_name"]
    layers_size = properties["layers_size"]
    activation_name = properties["activation"]
    criterion_name = properties["criterion"]
    activation = get_activation(activation_name)
    criterion = get_criterion(criterion_name)
    
    if data_name == "mnist":
        data_path = os.path.join(DATAPATH,"mnist","mlpnet")
        data = torch.load(os.path.join(data_path,"mnist_data.pth"))
        label = torch.load(os.path.join(data_path,"mnist_label.pth"))
    
    else:
        raise ValueError(f"{data_name} does not exist.")
    params = [data,label,layers_size]
    f = MLPNet(params,activation=activation,criterion=criterion)
    return f

def generate_cnn(properties):
    data_name = properties["data_name"]
    layers_size = properties["layers_size"]
    activation_name = properties["activation"]
    criterion_name = properties["criterion"]
    activation = get_activation(activation_name)
    criterion = get_criterion(criterion_name)

    if data_name == "mnist":
        data_path = os.path.join(DATAPATH,"mnist","cnn")
        data = torch.load(os.path.join(data_path,"images.pth"))
        label = torch.load(os.path.join(data_path,"labels.pth"))
        data_size = data.shape[2]
        print("data_size:",data_size)
        class_num = (torch.unique(label)).shape[0]
    
    params = [data,label,class_num,data_size,layers_size]
    f = CNNet(params,criterion=criterion,activation=activation)
    return f

def generate_polytope(properties):
    data_name = properties["data_name"]
    dim = properties["dim"]
    constraints_num = properties["constraints_num"]
    if data_name == "random":
        data_path = os.path.join(DATAPATH,"polytope")
        filename_A = f"A_{dim}_{constraints_num}.pth"
        b = torch.ones(constraints_num)
        if os.path.exists(os.path.join(data_path,filename_A)):
            A = torch.load(os.path.join(data_path,filename_A))
        else:
            A = torch.randn(constraints_num,dim)
            os.makedirs(data_path,exist_ok=True)
            torch.save(A,os.path.join(data_path,filename_A))
    
    params = [A,b]
    con= Polytope(params)
    return con

def generate_nonnegative(properties):
    dim = properties["dim"]
    con = NonNegative(params=[dim])
    return con

def generate_quadratic_constraints(properties):
    data_name = properties["data_name"]
    dim = properties["dim"]
    constraints_num = properties["constraints_num"]
    if data_name == "random":
        rank = dim//2
        data_path = os.path.join(DATAPATH,"quadratic_constraints")
        filename_Q = f"Q_{dim}_{rank}_{constraints_num}.pth"
        filename_b = f"b_{dim}_{rank}_{constraints_num}.pth"
        if os.path.exists(os.path.join(data_path,filename_Q)):
            Q = torch.load(os.path.join(data_path,filename_Q))
            b = torch.load(os.path.join(data_path,filename_b))
            c = -torch.ones(constraints_num)
        else:
            P = torch.randn(constraints_num,dim,dim)
            Q = torch.matmul(P,P.transpose(1,2))/dim
            b = torch.randn(constraints_num,dim)
            c = -torch.ones(constraints_num)
            os.makedirs(data_path,exist_ok=True)
            torch.save(Q,os.path.join(data_path,filename_Q))
            torch.save(b,os.path.join(data_path,filename_b))
    params = [Q,b,c]
    con = Quadratic(params)
    return con

def generate_fusedlasso(properties):
    a = float(properties["threshold1"])
    b = float(properties["threshold2"])
    params = [a,b]
    con = Fused_Lasso(params)
    return con

def generate_ball(properties):
    ord = properties["ord"]
    threshold = properties["threshold"]
    params = [threshold,ord]
    con = Ball(params)
    return con

def generate_huber(properties):
    delta = properties["delta"]
    threshold = properties["threshold"]
    params = [delta,threshold]
    con = Huber(params)
    return con

