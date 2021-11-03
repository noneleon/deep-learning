import numpy as np


def compute_error_for_line_given_points(b, w, points):
    '''

    :param b: 函数在坐标轴上的截距
    :param w: 函数的系数
    :param points: 输入一对数组(x,y)
    :return: 返回平均的错误率
    '''
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
        return totalError / float(len(points))

def step_gradient(b_new,w_new,points,learning_rate):
    '''

    :param b_new: 更新的偏执
    :param w_new: 更新的系数向量
    :param points: 向量组(x,y)
    :param learning_rate: 深度学习学习率
    :return: 更新后的偏执和系数(b,w)
    '''
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -2/N *(y-(w_new*x+b_new))
        w_gradient += -2/N*x*(y-(w_new*x+b_new))
    b = b_new - (learning_rate*b_gradient)
    w = w_new - (learning_rate*w_gradient)
    return [b,w]

def gradient_descent_runner(pionts,b,w,learning_rate,num_iterations):
    for i in range(num_iterations):
        b, w = step_gradient(b,w,np.array(points),learning_rate)
    return [b,w]

def train():
    points = np.genfromtxt('data.csv',delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 100
    print('staring gradient descend at b={0},w={1],error={2}'.format(compute_error_for_line_given_points(initial_b,initial_w,points)))
    print('runing.....')
    [b,m] = gradient_descent_runner(points,initial_b,initial_w,learning_rate, num_iterations)
    print('after {0} iteration b={1},m={2},error={3}'.format(num_iterations,b,w,compute_error_for_line_given_points(b,w,points)))

train()