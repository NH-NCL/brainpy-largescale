# brainpy-largescale

## 安装

```
pip install brainpy-largescale
```


## 引入库
```python
import brainpy as bp
import bpl
```

## 创建神经元簇

用简单但最典型的神经元模型 Leaky Integrate-and-Fire (LIF) 模型来构建兴奋性和抑制性神经元组

```python
a = bpl.LIF(
  200,
  V_rest=-60.,
  V_th=-50.,
  V_reset=-60.,
  tau=20.,
  tau_ref=5.)
b = bpl.LIF(
  100,
  V_rest=-60.,
  V_th=-50.,
  V_reset=-60.,
  tau=20.,
  tau_ref=5.)
```

## 创建突触
在定义 LIF 神经元组时，可以根据用户的需要调整参数。第一个参数表示神经元的数量。这里兴奋性和抑制性神经元的比例设置为 4:1。 V_rest 表示静息电位，V_th 表示触发阈值，V_reset 表示触发后的复位值，tau 为时间常数，tau_ref 为不应期的持续时间。方法是指在模拟中使用的数值积分方法。
那么这两组之间的突触连接可以定义如下：
```python
d = bpl.Exponential(
  a,
  b,
  bp.conn.FixedProb(0.04, seed=123),
  g_max=1.0,
  tau=8.0，
  delay_step=1)
```
在这里，我们使用指数突触模型 (bpl.Exponential) 来模拟突触连接。在模型的参数中，前两个分别表示突触前和突触后神经元组。第三个是指连接类型。在此示例中，我们使用 bp.conn.FixedProb，它以给定的概率将突触前神经元连接到突触后神经元（详细信息可在 Synaptic Connection 中获得）。以下三个参数描述了突触的动态特性，最后一个是 LIF 模型中的数值积分方法。

## 构建网络
定义完所有组件后，就可以组合成一个网络：
```python
net = bpl.respa.Network(a, b, d)
net.build()
```

## 增加输入

增加电流输入
```python
inputs = bpl.input_transform([(a, 20)])
```

## 增加脉冲监测
```python
monitor_spike = bpl.monitor_transform([a], attr='spike')
```

## 增加电压监测
```python
monitor_volt = bpl.monitor_transform([a], attr='V')
```

脉冲和电压监测合并到一个字典中：
```python
monitors = {}
monitors.update(monitor_spike)
monitors.update(monitor_volt)
```

## 增加脉冲和电压回调函数

```python
def spike(a: List[Tuple[int, float]]):
  print(a)

def volt(a: List[Tuple[int, float, float]]):
  # print(a)
  pass
```

## 运行网络
构建 SNN 后，我们可以将其用于动态模拟。要运行模拟，我们需要先将网络模型包装到运行器中,用户可以按如下方式初始DSRunner：

```python
runner = bpl.DSRunner(
  net,
  monitors=monitors,
  inputs=inputs,
  jit=False,
  spike_callback=spike,
  volt_callback=volt,
)
runner.run(10.)
```
其中调用函数接收模拟时间（通常以毫秒为单位）作为输入。 

##　可视化
```python
import matplotlib.pyplot as plt

if 'spike' in runner.mon:
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['spike'], show=True)
```

## What is brainpy-largescale?
Run [BrainPy](https://github.com/PKU-NIP-Lab/BrainPy) in multiple processes.

brainpy-largescale depends on [BrainPy](https://github.com/PKU-NIP-Lab/BrainPy) and [brainpy-lib](https://github.com/PKU-NIP-Lab/brainpylib), use the following instructions to [install brainpy package](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html).


## License<a id="quickstart"></a>
[Apache License, Version 2.0](https://github.com/NH-NCL/brainpy-largescale/blob/main/LICENSE)