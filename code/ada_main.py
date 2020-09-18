import numpy as np
import os
import matplotlib.pyplot as plt
## load data
#pickle data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# split data and labels
def load_CFIR10_btch(file_name):
    #### load per batch
    pic_file = unpickle(file_name)
    X = pic_file['data']
    Y = pic_file['labels']
    return X , Y

## reading each file of train and test
def load_CFIR10(f):
    Xs = []
    Ys = []
    for i in range(1,6):
        ef = os.path.join(f, 'data_batch_' + str(i))
        X,Y = load_CFIR10_btch(ef)
        Xs.append(X)
        Ys.append(Y)
    X_train = np.concatenate(Xs)
    Y_train = np.concatenate(Ys)
    test = os.path.join(f, 'test_batch')
    X_test,Y_test = load_CFIR10_btch(test)
    return X_train , Y_train, X_test,Y_test

def preparing_data(file_name):
    X_train, Y_train, X_test, Y_test = load_CFIR10(file_name)
    Y_test = np.array(Y_test)
    test_size = int(X_train.shape[0] / 10)
    test_slice, remainder = np.split(X_train.copy(), [test_size], axis=0)
    X_train = remainder
    #### standard X
    m = X_train.mean()
    v = X_train.std()
    X_train = (X_train - m) / v
    # X_test = (X_test - m) / v
    X_test = X_test/np.max(X_test)
    X_validation = test_slice
    X_validation = X_validation/np.max(X_validation)
    # X_validation = (X_validation - m) / v
    test_slice, remainder = np.split(Y_train.copy(), [test_size], axis=0)
    Y_train = remainder
    Y_validation = test_slice
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

#### initial parameters , by random values
def initialize_parameters_deep(layer_dimensions):
    parameters = {}
    # number of layers in the network
    L = len(layer_dimensions)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l-1], layer_dimensions[l ]) * 0.01
        parameters['b' + str(l)] = np.zeros((1,layer_dimensions[l]))
    return parameters

## activation functions
def sigmoid(Z):
    A = 1/ (1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    return A
#### derivative of activation function
def relu_prim(X):
    return 1. * (X > 0)

def sigmoid_prim(f):
    s = f*(1-f)
    return s
##### fedforward
def forward(X,parameters):
    #### number of layers
    L = len(parameters) // 2
    # at the first A is the input of model
    A = X
    ### definit cache to save parameters for backpropagation
    caches = []
    for l in range(1,L):
        Z = np.dot(A, parameters['W' + str(l)])+parameters['b'+str(l)]
        cache = (A, parameters['W' + str(l)], parameters['b' + str(l)])
        A = relu(Z)
        caches.append(cache)

    ZLast = np.dot(A, parameters['W' + str(L)])+parameters['b'+str(L)]
    cache = (A, parameters['W' + str(L)], parameters['b' + str(L)])
    # ALast = sigmoid(ZLast)
###### soft max
    exp_scores = np.exp(ZLast)
    ALast = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    caches.append(cache)
    return ALast , ZLast ,caches

############################################## one hot
def one_hot(y):
    onehot_y = np.zeros((len(y),10))
    for i in range(len(y)):
        onehot_y[i,y[i]] = 1
    return onehot_y

# comput cost
def compute_cost(ZL,Y):
    # number of examples

    cost = np.mean(Y * np.logaddexp(0,-ZL) + (1 - Y) * np.logaddexp(0,ZL))
    # To make sure our cost's shape is what we expect (e.g. this turns [[2]] into 2).
    cost = np.squeeze(cost)

    return cost

def cost_prim(AL,Y):
    dAL = (AL - Y)
    return dAL



###############################backprob
def model(AL,caches):
    L = len(caches)
    dc_dw = {}
    dc_db = {}
    df = cost_prim(AL,Y_train)
    m = (AL.shape)[0]

    for l in reversed(range(L)):
        A , W , b = caches[l]
        dc_dw['dc_dw' + str(l + 1)] =1./m * np.dot((A).T, df)
        dc_db['dc_db' + str(l + 1)] =1./ m * np.sum(df, axis=0, keepdims=True)
        df = np.dot(df, W.T) * relu_prim(A)
    return dc_dw , dc_db


def update_parameters(parameters,AL,caches):
    dc_dw, dc_db = model(AL,caches)
    learning_rate = 0.01
    L = len(parameters) // 2
    # print(L)# Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * dc_dw["dc_dw" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * dc_db["dc_db" + str(l + 1)]
    return parameters

def predict(X, parameters):
    # Forward propagation
    probas, ZL, caches = forward(X, parameters)  # convert probas to 0/1 predictions
    predicted = []
    for i in range(0, probas.shape[0]):
        j = np.squeeze(np.where(probas[i] == np.amax(probas[i])))
        predicted.append(j)
    return predicted
    # ALast = exp_scores /

def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
        # print(pred)
    return m

def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return (sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t)*100
X_train,Y_train,X_validation,Y_validation,X_test,Y_test = preparing_data('cifar-10-batches-py')
# X_train,Y_train,X_test,Y_test  = load_CFIR10('/content/drive/My Drive/cifar-10-batches-py')
expected = Y_train
Y_train = one_hot(Y_train)
expected_validation = Y_validation
Y_validation = one_hot(Y_validation)
expected_test = Y_test
Y_test = one_hot(Y_test)

def RMSprob(r,r_b, parameters, dc_dw, dc_db):
    beta = 0.999
    epsilon = 0.001
    delta = 1e-7
    L = len(parameters) // 2
    for l in range(L):
        r[l] = beta * r[l] + (1-beta) * np.sum(dc_dw["dc_dw" + str(l + 1)] ** 2)
        delta_theta = (-epsilon / np.sqrt(delta + r[l])) * dc_dw["dc_dw" + str(l + 1)]
        ########################## baias
        r_b[l] = beta * r_b[l] + (1 - beta) * np.sum(dc_db["dc_db" + str(l + 1)] ** 2)
        delta_theta_b = (-epsilon / np.sqrt(delta + r_b[l])) * dc_db["dc_db" + str(l + 1)]
        ########################### update data
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] + delta_theta
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] + delta_theta_b
    return parameters, r, r_b

def adagrad(r,r_b, parameters, dc_dw, dc_db):
    epsilon = 0.001
    delta = 1e-8
    # number of layer
    L = len(parameters) // 2
    for l in range(L):
        r[l] =  r[l] +  np.sum(dc_dw["dc_dw" + str(l + 1)] ** 2)
        delta_theta = (-epsilon / np.sqrt(delta + r[l])) * dc_dw["dc_dw" + str(l + 1)]
        # update of bias
        r_b[l] = r_b[l] + np.sum(dc_db["dc_db" + str(l + 1)] ** 2)
        delta_theta_b = (-epsilon / np.sqrt(delta + r_b[l])) * dc_db["dc_db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] + delta_theta
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] + delta_theta_b
    return parameters, r, r_b


def Adam(s, r, s_b, r_b, parameters, dc_dw, dc_db, i):
    beta1 = 0.99
    beta2 = 0.999
    epsilon = 0.001
    delta = 1e-8
    L = len(parameters) // 2
    for l in range(L):
        s[L - 1 - l] = beta1 * s[L - 1 - l] + (1 - beta1) * dc_dw["dc_dw" + str(l + 1)]
        r[l] =  beta2 * r[l] + (1 - beta2) * np.sum(dc_dw["dc_dw" + str(l + 1)] ** 2)
        s_hat = s[L - 1 - l] / (1 - beta1 ** i)
        r_hat = r[l] / (1 - beta2 ** i)
        delta_theta = (-epsilon * s_hat) / (np.sqrt(r_hat) + delta)
        ################################# bias
        s_b[L - 1 - l] =  beta1 * s_b[L - 1 - l] + (1 - beta1) * dc_db["dc_db" + str(l + 1)]
        r_b[l] =  beta2 * r_b[l] + (1 - beta2) * np.sum(dc_db["dc_db" + str(l + 1)] ** 2)
        s_hat = s_b[L - 1 - l] / (1 - beta1 ** i)
        r_hat = r_b[l] / (1 - beta2 ** i)
        delta_theta_b = (-epsilon * s_hat) / (np.sqrt(r_hat) + delta)
        ##################### update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] + delta_theta
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] + delta_theta_b
    return parameters, s, r, s_b, r_b

def adadelta(v,d,v_b,d_b,dc_dw, dc_db):
  beta = 0.95
  epsilon = 1e-8
  L = len(parameters) // 2
  for l in range(L):
    v[l] = beta * v[l] + (1-beta)*np.sum(dc_dw["dc_dw" + str(l + 1)] ** 2)
    delta_w = (- np.sqrt(d[l]+epsilon)/np.sqrt(v[l]+epsilon)) * dc_dw["dc_dw" + str(l + 1)]
    d[l] = beta * d[l] + (1-beta)* np.sum(delta_w ** 2)
    parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] + delta_w
    ##################### update bias
    v_b[l] = beta * v_b[l] + (1-beta)*np.sum(dc_db["dc_db" + str(l + 1)] ** 2)
    delta_b = (- np.sqrt(d_b[l]+epsilon)/np.sqrt(v_b[l]+epsilon)) * dc_db["dc_db" + str(l + 1)]
    d_b[l] = beta * d_b[l] + (1-beta)* np.sum(delta_b ** 2)
    parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] + delta_b
  return parameters, v, d, v_b, d_b

def train(iteration,):
    i = 0
    layers_dims = [3072,500,100,10]
    costs_train =[]
    costs_test = []
    costs_validation = []

    acuracy_train = []
    acuracy_test = []
    acuracy_validation = []
    parameters = initialize_parameters_deep(layers_dims)
    AL, ZL, caches = forward(X_train, parameters)
    dc_dw, dc_db = model(AL, caches)
    s = []
    for k in dc_dw:
        s.append(np.zeros(dc_dw[k].shape))
    s = np.array(s)
    r = np.zeros(len(layers_dims) - 1)
    s_b = []
    for k in dc_db:
        s_b.append(np.zeros(dc_db[k].shape))
    # s_b = np.array(s_b)
    r_b = np.zeros(len(layers_dims) - 1)
    d = np.zeros(len(layers_dims) - 1)
    d_b = np.zeros(len(layers_dims) - 1)
    while iteration != 0:
        i = i + 1
        AL,ZL,caches = forward(X_train, parameters)
        ac_train = np.mean(np.argmax(AL,axis=1)==expected)
        acuracy_train.append(ac_train)
        # predicted = predict(X_train,parameters)
        # conf = create_conf_matrix(expected,predicted,10)
        # ac = calc_accuracy(conf)
        A_validation, Z_validation, caches_validation = forward(X_validation, parameters)
        ac_validation = np.mean(np.argmax(A_validation, axis=1) == expected_validation)
        acuracy_validation .append(ac_validation)
        A_test, Z_test, caches_test = forward(X_test, parameters)
        ac_test = np.mean(np.argmax(A_test, axis=1) == expected_test)
        acuracy_test.append(ac_test)
        print('accuracy of tain is',ac_train)
        print('accuracy of validation is', ac_validation)
        print('accuracy of test is', ac_test)
        # parameters= update_parameters(parameters,AL,caches)
        # parameters, s, r, s_b, r_b =  Adam(s, r, s_b, r_b, parameters, dc_dw, dc_db, i)
        # parameters, r, r_b = RMSprob(r,r_b, parameters, dc_dw, dc_db)
        parameters, r, r_b = adagrad(r,r_b, parameters, dc_dw, dc_db)
        # parameters, r, d, r_b, d_b = adadelta(r,d,r_b,d_b,dc_dw, dc_db)
        loss_train = compute_cost(ZL,Y_train)
        loss_validation = compute_cost(Z_validation, Y_validation)
        loss_test = compute_cost(Z_test, Y_test)
        print("Loss after iteration %i: %f" % (i, loss_train))
        print("Loss after iteration %i: %f" % (i, loss_validation))
        print("Loss after iteration %i: %f" % (i, loss_test))
        costs_validation.append(loss_validation)
        costs_train.append(loss_train)
        costs_test.append(loss_test)
        iteration = iteration - 1

    return parameters, costs_train,costs_validation,costs_test,acuracy_train,acuracy_validation,acuracy_test

parameters, costs_train,costs_validation,costs_test,acuracy_train,acuracy_validation,acuracy_test = train(50)
fig = plt.figure(figsize=(5,5))
plt.plot(costs_validation,'g')
plt.plot(costs_train,'b')
plt.plot(costs_test,'r')
fig = plt.figure(figsize=(5,5))
plt.plot(acuracy_train,'b')
plt.plot(acuracy_validation,'g')
plt.plot(acuracy_test,'r')
plt.show()
