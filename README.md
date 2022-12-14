# brainpy-largescale

## Install
Only support `Linux`
```
pip install brainpy-largescale
```


## import
```python
import brainpy as bp
import bpl
```

## set platform
```
bpl.set_platform('cpu')
```
only support cpu.

## create population

Use Leaky Integrate-and-Fire (LIF)

```python
a = bpl.neurons.LIF(300, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
b = bpl.neurons.LIF(100, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
```

## create synapse
```python
d = bpl.synapses.Exponential(a, b, bp.conn.FixedProb(0.4, seed=123), g_max=10, tau=5., delay_step=1)
```

## construct network

```python
net = bpl.Network(a, b, d)
net.build()
```

## add input

add current input
```python
inputs = [bpl.device.Input(a, 20), bpl.device.Input(b, 10)]
```

## add spike monitor
```python
monitor_spike = bpl.device.Monitor([a, b], bpl.device.MonitorKey.spike)
```

## add volt monitor
```python
monitor_volt = bpl.device.Monitor([b], bpl.device.MonitorKey.volt)
```

```python
monitors = [monitor_spike, monitor_volt]
```

## add spike and volt callback

```python
def spike(a: List[Tuple[int, float]]):
  if a:
    print(a)


def volt(a: List[Tuple[int, float, float]]):
  # print(a)
  pass
```

## Run

```python
runner = bpl.runner.DSRunner(
  net,
  monitors=monitors,
  inputs=inputs,
  jit=False,
  spike_callback=spike,
  volt_callback=volt,
)
runner.run(10.)
```
 

##　visualization
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