#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Acknowledgement: twmeggs' implementation of ANFIS in Python was very
    useful in understanding how the ANFIS structures could be interpreted:
        https://github.com/twmeggs/anfis
'''

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F


dtype = torch.float

"""
    @FuzzifyVariable(torch.nn.Module)
    类成员变量:
        - mfdefs: 单个隶属函数/隶属函数列表
        - padding:
"""
class FuzzifyVariable(torch.nn.Module):
    """
        初始化FuzzifyVariable类,
        初始化参数为
            - 隶属函数类(mfdefs = mfdef)
            - 隶属函数类列表(mfdefs = [mfdef, ...])
    """
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        # 判断mfdefs是否为类列表
        if isinstance(mfdefs, list): 
            # mfnames = ['mf0', 'mf1', ...]
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            """
                mfdefs为有顺序的字典
                mfdefs = OrderedDict(
                    [
                        ('mf0', [GaussMembFunc(), ...]),
                        ('mf1', [GaussMembFunc(), ...]),
                    ]
                )
            """
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        # 转为模型可识别的类型
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    """
        @members(self)
        返回可迭代的元组
        return = odict_items(
            [
                ('mf0', [GaussMembFunc(), ...]),
                ('mf1', [GaussMembFunc(), ...]),
            ]
        )
    """
    def members(self):
        return self.mfdefs.items()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    """
        @fuzzify(self, x)
        为这些输入值生成(函数名, 模糊值)列表。
    """
    def fuzzify(self, x):
        for mfname, mfdef in self.mfdefs.items():
            # 获取模糊值, 将x带入隶属函数中进行求解
            yvals = mfdef(x)
            yield(mfname, yvals)

    """
        @forward(self, x)
        返回模糊值张量
        x.shape: n_cases
        y.shape: n_cases * n_mfs
    """
    def forward(self, x):
        # 利用隶属函数计算模糊值, 结果进行横向拼装
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in,\
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


"""
    @AntecedentLayer(torch.nn.Module)
    类成员变量:
        - mf_indices: 第二层触发强度的乘积索引值
"""
class AntecedentLayer(torch.nn.Module):
    """
        初始化第二层与第三层网络
        初始化参数为:
            - varlist: FuzzifyVariable类列表
    """
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        # 计算隶属函数的个数
        mf_count = [var.num_mfs for var in varlist]
        """
            @itertools.product
            Input : arr1 = [1, 2, 3] 
                    arr2 = [5, 6, 7] 
            Output : [(1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)] 
            生成第二层触发强度的乘积索引值
        """
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        """
            假定invardefs = [
                    ['x0', ['f1', 'f2', 'f3']],
                    ['x1', ['f4', 'f5']],
                ]
            则self.mf_indices = tensor([[0, 0],
                                        [0, 1],
                                        [1, 0],
                                        [1, 1],
                                        [2, 0],
                                        [2, 1]])
            [0, 0]表示x0的第一个模糊值与x1的第一个模糊值
            mf_indices.shape = n_rules * n_in
        """
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self):
        return len(self.mf_indices)

    """
        @extra_repr(self, varlist=None):
        触发强度乘积索引值更加精确的描述:
        invardefs = [
            ['x0', ['f1', 'f2', 'f3']],
            ['x1', ['f4', 'f5']],
        ]
        则返回为
            x0 is f1 and x1 is f4
            x0 is f1 and x1 is f5
            x0 is f2 and x1 is f4
            ...
    """
    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # 重复规则索引以等于批量大小：
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # 使用索引来填充规则前提
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''
    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}'\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)

"""
    @AnfisNet(torch.nn.Module)
    类网络成员变量:
        - description: 描述语句
        - outvarnames: 网络输出结果
        - hybrid: 是否使用混合最小二乘法
        - num_in: 网络输入变量个数
        - num_rules: 触发强度的个数/第二层元素的个数
        - layer: 网络的所有层
"""
class AnfisNet(torch.nn.Module):
    """
        ANFIS网络初始化部分
        网络输入参数:
            - description: 描述语句, 如'Simple classifier'
            - invardefs: 输入变量与隶属函数列表, 如: invardefs = [['x0', [GaussMembFunc(), ...]], ...]
            - outvarnames: 网络输出结果, 如: outvarsnames = ['y0', 'y1', ...]
            - hybrid: 是否使用混合最小二乘法
    """
    def __init__(self, description, invardefs, outvarnames, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        # 获取输入变量名称列表 varnames = ['x0', 'x1', ...]
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        # 计算网络输入变量个数
        self.num_in = len(invardefs)
        """
            计算触发强度的个数
            获取每个输入变量隶属函数的个数，形成列表，再计算个数的乘积:
            invardefs = [
                ['x0', ['f1', 'f2', 'f3']],
                ['x1', ['f4', 'f5']],
            ]
            self.num_rules = 6
        """
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            # 第一层
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            # 第二三层
            ('rules', AntecedentLayer(mfdefs)),
            # normalisation layer is just implemented as a function.
            # 第四层
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
            ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    """
        获取IF-THEN准则
    """
    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        # 计算两个tensor的矩阵乘法
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# These hooks are handy for debugging:

def module_hook(label):
    ''' Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
