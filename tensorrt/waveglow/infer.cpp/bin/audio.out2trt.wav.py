from scipy.io.wavfile import write
import numpy as np

l = []
with open('audio.out', 'r') as f:
    n = f.readline()
    while n != '':
        l.append(float(n))
        n = f.readline()

l = np.array(l)
l = l * 32756; audio = l.astype('int16')
write('trt.wav', 22050, audio)
print('original conditioner data in cond.in feed into the engine and got audio.out as output, turned to trt.wav sucessfully.')
