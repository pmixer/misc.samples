# WaveGlow Inference Optimization using TensorRT

Pls make sure you have Nvidia GPU enpowered machine with TensorRT(5.1 used in my experiments) installed.
We used Python 3 for generating the TensorRT engine with pytorch(1.0 cpu version) and tensorrt packages installed.

Pls download WaveGlow model by clicking the link in `trans.py/README.md` and put it inside `trans.py` folder.

For trans the model from pytorch to tensorrt pls do:

```
cd trans.py
python pt2engine.py
```

engines will be saved as `trans.py/*.engine`, then pls:

```
cd infer.cpp
make
cd bin
sampleWaveGlow [Engine Path]
```

to use generated engine for inference using default mel.
