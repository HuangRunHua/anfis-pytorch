# 自适应网络模糊推理系统(ANFIS)Pytorch源码浅析

> **标注**
>
> 本文主要分析由[James Power](http://www.cs.nuim.ie/~jpower/)开源于Github上的[anfis-pytorch](https://github.com/jfpower/anfis-pytorch)代码。
>
> 关于ANFIS的基础知识参考[浅析自适应网络模糊推理系统(ANFIS)](https://huangrunhua.github.io/2022/05/16/Talk-About-ANFIS/)

## 代码框架分析
ANFIS框架主要分为三个文件：
- `anfis.py` 定义ANFIS网络核心的部分。
- `membership.py` 该文件下定义了各种各样的隶属函数。
- `Experimental.py` 训练和测试FIS以及绘制一些图表等的实验基础设施。

程序包括一些可执行的文件
- `jang_examples.py` [Jyh-Shing Roger Jang](http://mirlab.org/jang/) 于1993年发表的开创性论文 [ANFIS: adaptive-network-based fuzzy inference system](https://ieeexplore.ieee.org/abstract/document/256541) 的仿真与实现。

## 隶属函数分析(membership.py)
本文件下定义了几类隶属函数的类模型，包括高斯隶属函数类模型(`GaussMembFunc`)与贝尔隶属函数类模型(`BellMembFunc`)等。本文主要分析高斯隶属函数类模型。

#### GaussMembFunc类模型
```python
class GaussMembFunc(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)
```

上述代码定义了一个高斯隶属函数类模型，类继承于`torch.nn.Module`。`GaussMembFunc`的初始化需要传入两个参数均值`mu`与标准差`sigma`，这是由于若已知高斯函数均值与标准差，该分布便可以被确定下来。在初始化函数`__init__()`中存在如下两行代码:
```python
self.register_parameter('mu', _mk_param(mu))
self.register_parameter('sigma', _mk_param(sigma))
```

`register_parameter()`将一个不可训练的类型`Tensor`转换成可以训练的类型参数，并将参数与模型绑定，相当于变成了模型的一部分，成为了模型中可以根据训练进行变化的参数。注意到高斯隶属函数中均值与标准差均为前提参数，将会使用梯度下降法来进行数值更新。

`_mk_param()`函数的定义如下:
```python
def _mk_param(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))
```
该函数的作用在于将标量值转变为一个高斯隶属函数类模型可识别的torch参数。

#### make_gauss_mfs函数
代码文件内还定义了一个`make_gauss_mfs`函数，用于获得多高斯隶属函数模型的情况:
```python
def make_gauss_mfs(sigma, mu_list):
    return [GaussMembFunc(mu, sigma) for mu in mu_list]
```

`make_gauss_mfs`函数参数为隶属函数的标准差(`sigma`)与均值列表(`mu_list`)，函数返回一个高斯隶属函数类模型列表。

#### make_anfis函数
当获得到隶属函数之后，便可以将隶属函数与输入变量添加到ANFIS网络同时初始化网络，`make_anfis`函数的作用便在于此。完整的`make_anfis`函数代码如下:
```python
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
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
    outvars = ['y{}'.format(i) for i in range(num_out)]

    # 将 invars 和 outvars 作为参数传入 AnfisNet() 建立 ANFIS 网络
    
    model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)
    return model
```

`make_anfis`函数的参数为网络的输入变量`x`，隶属函数的个数`num_mfs`，网络输出量的个数`num_out`与是否使用最小二乘混合方法的标识位`hybrid`。默认情况下隶属函数个数为5个，网络为单个输出，同时网络默认使用最小二乘混合方法。

变量`invars`是存放隶属函数的标签与隶属函数类模型的列表，`invars`通过如下代码添加变量:
```python
invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
```
执行之后，`invars`列表被修改为如下形式:
```python
invars = [
            ['x0', [GaussMembFunc(), ...]], 
            ['x1', [GaussMembFunc(), ...]], ...
         ]
```

变量`outvars`是存放网络输出变量标签的列表，其内部形式为:
```python
outvars = ['y0', 'y1', ...]
```

`make_anfis`函数的输出为ANFIS网络，网络通过如下代码创建:
```python
model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)
```

## 网络层分析(anfis.py)
该文件下定义了ANFIS网络的层结构。

#### AnfisNet(torch.nn.Module)类模型
`AnfisNet(torch.nn.Module)`定义了一个完整的自适应模糊推理。其类初始化方法为:
```python
def __init__(self, description, invardefs, outvarnames, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            ('consequent', cl),
            ]))
```
类初始化接受的参数一共有四个：
- `description`: 描述语句, 如'Simple classifier'
- `invardefs`: 输入变量与隶属函数列表, 如: ```invardefs = [['x0', [GaussMembFunc(), ...]], ...]```
- `outvarnames`: 网络输出结果, 如: ```outvarsnames = ['y0', 'y1', ...]```
- `hybrid`: 是否使用混合最小二乘法

`AnfisNet`类网络成员变量定义如下:
- `description`: 描述语句
- `outvarnames`: 网络输出变量名
- `hybrid`: 是否使用混合最小二乘法
- `num_in`: 网络输入变量个数
- `num_rules`: 触发强度的个数/第二层元素的个数
- `layer`: 网络的所有层

初始化阶段将传入的输入变量与隶属函数列表`invardefs`进行处理，设
```python
`invardefs` = [
    ['x0', [mf(), mf(), ...]],
    ['x1', [mf(), mf(), ...]],
    ...
]
```
则先将变量名提取出来:
```python
varnames = [v for v, _ in invardefs]
```
再将隶属函数传入模糊变量类模型内形成列表:
```python
mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
```
网络输入变量名`varnames`与模糊变量类模型列表`mfdefs`作为输入变量传入网络第一层`FuzzifyLayer`与第二层`AntecedentLayer`:
```python
FuzzifyLayer(mfdefs, varnames)
AntecedentLayer(mfdefs)
```

由于ANFIS网络中第二层与第三层之间需要进行触发强度的计算，多个输入变量的隶属函数两两之间需要进行乘法运算，因此类成员变量`num_rules`存储总共需要计算的次数:
```python
self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
```

例如，
```python
invardefs = [
    ['x0', ['f1', 'f2', 'f3']],
    ['x1', ['f4', 'f5']],
]
```
则`self.num_rules = 6`

类成员变量`self.hybrid`用于判断网络是否使用混合最小二乘法，若使用则网络第四层为`ConsequentLayer`，否则为`PlainConsequentLayer`.

类成员变量`self.layer`定义了完整的网络内部层结构，为顺序的字典，可以通过键来进行索引。
```python
self.layer = torch.nn.ModuleDict(OrderedDict([
    ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
    ('rules', AntecedentLayer(mfdefs)),
    ('consequent', cl),
    ]))
```

第四层网络的系数可以通过如下方式获取:
```python
self.layer['consequent'].coeff
```

由于第四层网络需要使用前向传播或混合最小二乘法来更新，因此ANFIS网络中定义了两个类方法来实现系数的更新:
```python
@coeff.setter
def coeff(self, new_coeff):
    self.layer['consequent'].coeff = new_coeff

def fit_coeff(self, x, y_actual):
    if self.hybrid:
        self(x)
        self.layer['consequent'].fit_coeff(x, self.weights, y_actual)
```

ANFIS网络的前向传播部分如下:
```python
def forward(self, x):
    self.fuzzified = self.layer['fuzzify'](x)
    self.raw_weights = self.layer['rules'](self.fuzzified)
    self.weights = F.normalize(self.raw_weights, p=1, dim=1)
    self.rule_tsk = self.layer['consequent'](x)
    y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
    self.y_pred = y_pred.squeeze(2)
    return self.y_pred
```

#### FuzzifyVariable(torch.nn.Module)类模型
`FuzzifyVariable(torch.nn.Module)类`定义了模糊化的基础操作。其类初始化方法为:
```python
# mfdefs的类型为GaussMembFunc或[GaussMembFunc, GaussMembFunc]

def __init__(self, mfdefs):
    super(FuzzifyVariable, self).__init__()
    # 判断mfdefs是否为隶属函数类列表
    
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
```

类成员变量为:
- `mfdefs`: 单个隶属函数/隶属函数列表
- `padding`

前向传播函数为:
```python
def forward(self, x):
    # 利用隶属函数计算模糊值, 结果进行横向拼装
    
    y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
    if self.padding > 0:
        y_pred = torch.cat([y_pred,
                            torch.zeros(x.shape[0], self.padding)], dim=1)
    return y_pred
```

在ANFIS网络中，`FuzzifyVariable`类模型作为输入变量传入`FuzzifyLayer`模糊层内。

#### FuzzifyLayer(torch.nn.Module)模糊层
`FuzzifyLayer(torch.nn.Module)`类定义了ANFIS的第一层网络。其初始化过程接收两个参数:
- `varmfs`: `FuzzifyVariable`类模型/模型列表
- `varnames`: 变量名列表

若没有变量名则在初始化过程中自动创建变量名列表作为类成员变量`self.varnames`。
```python
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
```

第一层最终输出为各个输入变量在各自隶属函数的模糊下的结果，前向传播函数主要将来自`FuzzifyVariable`类的输出结果进行整合堆栈，用于下一层网络使用:
```python
def forward(self, x):
    assert x.shape[1] == self.num_in,\
        '{} is wrong no. of input values'.format(self.num_in)
    y_pred = torch.stack([var(x[:, i:i+1])
                            for i, var in enumerate(self.varmfs.values())],
                            dim=1)
    return y_pred
```

#### AntecedentLayer(torch.nn.Module)类模型(第二层网络)
类模型`AntecedentLayer(torch.nn.Module)`定义了ANFIS中的第二层网络，用于计算触发强度。该类接收的初始化参数为`FuzzifyVariable`类列表。初始化函数如下:
```python
def __init__(self, varlist):
    super(AntecedentLayer, self).__init__()
    mf_count = [var.num_mfs for var in varlist]
    mf_indices = itertools.product(*[range(n) for n in mf_count])
    self.mf_indices = torch.tensor(list(mf_indices))
```
`mf_indices`用于存储触发强度的乘积索引值，例如:
```python
Input : arr1 = [1, 2, 3] 
        arr2 = [5, 6, 7] 
Output : mf_indices = [(1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)] 
```
类成员变量`self.mf_indices`是一个存储触发强度的乘积索引值的张量, 例如:
```python
invardefs = [
    ['x0', ['f1', 'f2', 'f3']],
    ['x1', ['f4', 'f5']],
]

self.mf_indices = tensor([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1],
                          [2, 0],
                          [2, 1]])
# 其中[0, 0]表示x0的第一个模糊值与x1的第一个模糊值
```

该类前向传播返回触发强度乘积结果:
```python
def forward(self, x):
    # 重复规则索引以等于批量大小：
    
    batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
    # 使用索引来填充规则前提
    
    ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
    # ants.shape is n_cases * n_rules * n_in
    
    # Last, take the AND (= product) for each rule-antecedent
    
    rules = torch.prod(ants, dim=2)
    return rules
```

#### ConsequentLayer(torch.nn.Module)第四层网络
类模型`ConsequentLayer(torch.nn.Module)`定义了存在混合最小二乘法的第四层网络，其中`fit_coeff(self, x, weights, y_actual)`函数实现了最小二乘更新结果参数的功能:
```python
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
        coeff_2d, _ = torch.gels(y_actual_2d, weighted_x_2d)
    except RuntimeError as e:
        print('Internal error in gels', e)
        print('Weights are:', weighted_x)
        raise e
    coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
    # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
    
    self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1).transpose(1, 2)
    # coeff dim is thus: n_rules * n_out * (n_in+1)
```
前向传播主要实现输入特征的线性组合（参数称为结果参数，结果参数通过最小二乘法进行数值更新）与归一化触发强度进行乘积运算：
```python
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
```

#### WeightedSumLayer(torch.nn.Module)去模糊化层
类模型`WeightedSumLayer`定义了第五层去模糊化层网络，前向传播返回第四层网络加权求和的结果，该类定义如下:
```python
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
```

## 网络训练模块(experimental.py)
文档`experimental.py`定义了ANFIS训练过程与测试过程，还包括Pytorch官网教程中的的2层线性神经网络。

#### train_anfis_with(model,...)函数
`train_anfis_with(model,...)`函数给出了ANFIS训练过程的所有细节。该函数接收参数为:
- `model`: 网络模型 
- `data`: 训练数据
- `optimizer`: 优化器的选择
- `criterion`: 损失函数
- `epochs`: 迭代次数
- `show_plots`: 是否展示训练过程图表

网络训练过程具体细节如代码所示:
```python
def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
    
        Train the given model using the given (x,y) data.
        
    '''
    errors = []  # Keep a list of these for plotting afterwards
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%'
                  .format(t, mse, rmse, perc_loss))
    # End of training, so graph the results:
    
    if show_plots:
        plotErrors(errors)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)
```

优化器`optimizer`与损失函数`criterion`定义在`train_anfis(model,...)`内。

#### train_anfis(model,...)函数
```python
def train_anfis(model, data, epochs=500, show_plots=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)
```

#### test_anfis(model,...)函数
函数`test_anfis(model,...)`给出了ANFIS测试细节，函数源代码如下:
```python
def test_anfis(model, data, show_plots=False):
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)
```

## 网络实战(jang_examples.py)
文档`jang_examples.py`给出了Jyh-Shing Roger Jang于1993年发表的开创性论文[ANFIS: adaptive-network-based fuzzy inference system](https://ieeexplore.ieee.org/abstract/document/256541)的复现结果。

本文仅分析第一个示例。

#### 示例: 两输入非线性方程的建模
真实非线性方程定义如下:

$$
z = sinc(x, y) = \frac{sin(x)}{x} \times \frac{sin(y)}{y}
$$

该函数的实现过程如下:
```python
def sinc(x, y):
    def s(z):
        return (1 if z == 0 else np.sin(z) / z)
    return s(x) * s(y)
```

训练数据集的生成通过函数`make_sinc_xy`来实现:
```python
def make_sinc_xy(batch_size=1024):
    pts = torch.arange(-10, 11, 2)
    x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
    y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
    td = TensorDataset(x, y)
    return DataLoader(td, batch_size=batch_size, shuffle=True)
```

函数执行过程中, 中间变量`pts`为tensor，将区间(-10, 10)按照步长为2划分，作为变量的取值范围:
```python
pts = tensor([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
```

由于z函数两个自变量的区间相同，因此`pts`同时代表两个自变量的取值范围。自变量x与y两两组合形成变量对(x, y)作为变量带入函数`sinc(x, y)`内生成网络输出数据，变量对一共121对，通过如下语句生成:
```python
>>> x = torch.tensor(list(itertools.product(pts, pts)), dtype=dtype)
>>> x = tensor([[-10., -10.],
                [-10.,  -8.],
                [-10.,  -6.],
                [-10.,  -4.],
                [-10.,  -2.],...])
```

> **注意**
>
> 注意程序内的xy与数学公式上的xy的意义不同。

将上述变量区间与代入公式计算后的结果作为输入与输出传入`TensorDataset`并利用`DataLoader`转为可训练的数据类型。

示例1为两输入一输出的过程，该模型定义如下:
```python
def ex1_model():
    invardefs = [
            ('x0', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ('x1', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ]
    outvars = ['y0']
    anf = anfis.AnfisNet('Jang\'s example 1', invardefs, outvars)
    return anf
```

主函数部分的调用过程如下:
```python
model = ex1_model()
train_data = make_sinc_xy()
train_anfis(model, train_data, 20, show_plots)
```
