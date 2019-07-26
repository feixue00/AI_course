import numpy as np
import random
import matplotlib.pyplot as plt

#给直线套一个马甲
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
#总的样本集，训练的时候，每次迭代从样本集中随机抽取几个数据
def get_sample(samples=500):
#    w = np.random.randint(10,size=samples)
#    b = np.random.randint(10,size=samples)
    shape=(samples,2)
    data_list = np.zeros(shape)
    w = np.random.uniform(0,5)
    b = np.random.uniform(0,3)
    data_list[:,0] = np.random.rand(samples)*100
    data_list[:,1] = sigmoid(w * data_list[:,0] + b + np.random.rand(samples)*np.random.randint(-1,1))
    print('w_actural:{0}, b_actural:{1}'.format(w, b))
    return data_list,w,b

def step_gredient(patch_list,w_iteration,b_iteration,lr):
    x_list = patch_list[:,0]
    y_list = patch_list[:,1]
    y_hypothesis = sigmoid(w_iteration * x_list + b_iteration)
    batch_size = len(x_list)
    b_gradient = (np.sum(y_hypothesis-y_list))/batch_size
    w_gradient = ( np.sum( np.multiply((y_hypothesis-y_list),x_list) ) )/batch_size
    w_iteration -= lr*w_gradient
    b_iteration -= lr*b_gradient
    return w_iteration,b_iteration
    
def train(data_list,batch_size,max_iter,lr):
    w_iteration = 0
    b_iteration = 0
    for i in range(max_iter):
        np.random.shuffle(data_list)#先洗乱再取前batch_size个数据，相当于随机取值
        batch = data_list[:batch_size,:]
        w_iteration,b_iteration = step_gredient(batch,w_iteration,b_iteration,lr)
    print('w_iteration:{0}, b_iteration:{1}'.format(w_iteration, b_iteration))
    
    
def run():
    data_list,w,b = get_sample(2000)
    train(data_list,batch_size=200,max_iter=20000,lr=0.1)
    
if __name__=='__main__':
  run()

    
