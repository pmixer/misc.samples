this is a python extension sample to apply mergesort(with buffer for accelerate execution and save memory) for counting reverse pair number in 1D numpy array, compile by:

```sh
make all
```

then counter reverse pair number by

```python
import counter
import numpy as np
arr = np.random.random(6)
buff = np.zeros(6)
rev_pair_num = counter.inversion_count(arr, buff)
```
