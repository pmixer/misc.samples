# WaveGlow Inference Optimization using TensorRT

Pls make sure you have Nvidia GPU enpowered machine with pytorch(1.0 cpu version) and tensorrt (5.1 used in experiments) setup(in Python 3.X).

Pls download WaveGlow model by clicking the link in `trans.py/README.md` and put it inside `trans.py` folder.

For trans the model from pytorch to tensorrt pls do:

```
cd trans.py
python pt2engine.py
```

engines could be saved as `trans.py/*.engine`, then pls do:

```
cd online_infer.cpp
make
cd bin
sampleWaveGlow [Engine Path]
```

to use generated engine for inference as a demo of online serving.
