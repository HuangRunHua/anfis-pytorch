#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import torch

from anfis import AnfisNet

"""
    将标量值转变为一个高斯隶属函数类模型可识别的torch参数
"""
def _mk_param(val):
    if isinstance(val, torch.Tensor):
        # 获取可迭代的数据
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))

"""
    高斯隶属函数类模型
    初始化需要传入两个参数均值mu与标准差sigma
"""
class GaussMembFunc(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        """
            将一个不可训练的类型Tensor转换成可以训练的类型参数
            将参数与模型绑定, 相当于变成了模型的一部分
            成为了模型中可以根据训练进行变化的参数
        """
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)

"""
    make_gauss_mfs函数参数:
        - 隶属函数的标准差(sigma)
        - 均值列表(mu_list)
    函数返回:
        - 高斯隶属函数类模型列表
"""
def make_gauss_mfs(sigma, mu_list):
    '''Return a list of gaussian mfs, same sigma, list of means'''
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c) for c in clist]


class TriangularMembFunc(torch.nn.Module):
    '''
        Triangular membership function; defined by three parameters:
            a, left foot, mu(x) = 0
            b, midpoint, mu(x) = 1
            c, right foot, mu(x) = 0
    '''
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        assert a <= b and b <= c,\
            'Triangular parameters: must have a <= b <= c.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    @staticmethod
    def isosceles(width, center):
        '''
            Construct a triangle MF with given width-of-base and center
        '''
        return TriangularMembFunc(center-width, center, center+width)

    def forward(self, x):
        return torch.where(
            torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                torch.ByteTensor(self.b < x) & torch.ByteTensor(x <= self.c),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x, requires_grad=True)))

    def pretty(self):
        return 'TriangularMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_tri_mfs(width, clist):
    '''Return a list of triangular mfs, same width, list of centers'''
    return [TriangularMembFunc(c-width/2, c, c+width/2) for c in clist]


class TrapezoidalMembFunc(torch.nn.Module):
    '''
        Trapezoidal membership function; defined by four parameters.
        Membership is defined as:
            to the left of a: always 0
            from a to b: slopes from 0 up to 1
            from b to c: always 1
            from c to d: slopes from 1 down to 0
            to the right of d: always 0
    '''
    def __init__(self, a, b, c, d):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b and b <= c and c <= d,\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.register_parameter('d', _mk_param(d))

    @staticmethod
    def symmetric(topwidth, slope, midpt):
        '''
            Make a (symmetric) trapezoid mf, given
                topwidth: length of top (when mu == 1)
                slope: extra length at either side for bottom
                midpt: center point of trapezoid
        '''
        b = midpt - topwidth / 2
        c = midpt + topwidth / 2
        return TrapezoidalMembFunc(b - slope, b, c, c + slope)

    @staticmethod
    def rectangle(left, right):
        '''
            Make a Trapezoidal MF with vertical sides (so a==b and c==d)
        '''
        return TrapezoidalMembFunc(left, left, right, right)

    @staticmethod
    def triangle(left, midpt, right):
        '''
            Make a triangle-shaped MF as a special case of a Trapezoidal MF.
            Note: this may revert to general trapezoid under learning.
        '''
        return TrapezoidalMembFunc(left, midpt, midpt, right)

    def forward(self, x):
        yvals = torch.zeros_like(x)
        if self.a < self.b:
            incr = torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b)
            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
        if self.b < self.c:
            decr = torch.ByteTensor(self.b < x) & torch.ByteTensor(x < self.c)
            yvals[decr] = 1
        if self.c < self.d:
            decr = torch.ByteTensor(self.c <= x) & torch.ByteTensor(x < self.d)
            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
        return yvals

    def pretty(self):
        return 'TrapezoidalMembFunc a={} b={} c={} d={}'.\
            format(self.a, self.b, self.c, self.d)


def make_trap_mfs(width, slope, clist):
    '''Return a list of symmetric Trap mfs, same (w,s), list of centers'''
    return [TrapezoidalMembFunc.symmetric(width, slope, c) for c in clist]


# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}

"""
    make_anfis函数的参数:
        - 网络的输入变量x
        - 隶属函数的个数num_mfs
        - 网络输出量的个数num_out
        - 是否使用最小二乘混合方法的标识位hybrid
"""
def make_anfis(x, num_mfs=5, num_out=1, hybrid=True):
    # 获取输入量的个数
    num_invars = x.shape[1]
    # 沿着x每列求最大值和最小值
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    # 得到输入各个状态量的取值范围
    ranges = maxvals-minvals
    invars = []
    for i in range(num_invars):
        # 计算高斯隶属函数的方差
        sigma = ranges[i] / num_mfs
        mulist = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()
        """
            invars = [
            ['x0', [GaussMembFunc(), ...]], 
            ['x1', [GaussMembFunc(), ...]], ...
         ]
        """
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
    # outvars = ['y0', 'y1, ...]
    outvars = ['y{}'.format(i) for i in range(num_out)]
    # 将 invars 和 outvars 作为参数传入 AnfisNet() 建立 ANFIS 网络
    model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)
    return model
