import random
import numpy as np
from pyo import *

s = Server().boot().start()

# Granulator
snd = SndTable("guitar1.wav").normalize()
position = SigTo(0.5, time=1.75, init=0.5, mul=snd.getSize(), add=Noise(5))
env = WinTable(2)
duration = SigTo(1, time=0.1)
pitch = SigTo(1, time=0.1)
volume = SigTo(0.4, time=0.1)
grn = Granulator(table=snd, env=env, pitch=pitch, pos=position, dur=duration, grains=2, basedur=1, mul=volume)
def rand_point():
    position.value = random.random()
pat = Pattern(rand_point, time=1.75).play()

# Lowpass Filter

leftRightAmp = SigTo([1, 1], time=0.1)

cutoff = SigTo(8000, time=0.05, mul=1, add=1)
srscale = SigTo(1, time=0.1)
deg = Degrade(grn, bitdepth=32, srscale=srscale, mul=1)
low_pass = MoogLP(deg, cutoff, mul=1)

freq = SigTo(400, time=0.05, mul=1)
noise_vol = SigTo(0.3, time=0.05, mul=1)
q = SigTo(20, time=0.05, mul=1)
band_pass = Resonx(PinkNoise(0.3), freq=freq, q=q, mul=noise_vol, stages=2)

comp = Compress(low_pass + band_pass, thresh=-20, ratio=4, risetime=0.005, falltime=0.10, knee=0.5, mul=leftRightAmp).out()

# ----- Receiver -----
def getDataMessage(address, *args):

    blink = args[0]
    x = args[1]
    y = args[2]
    z = args[3]
    mouth_horiz = args[4]
    mouth_vert = args[5]
    eyebrows = args[6]
    
    volume.setValue((np.interp(z, [0, 100], [0, 4])).item())

    if blink == 0:
        pat.time = 1.75

        l, r = 1.0 - (x / 100), (x / 100)
        leftRightAmp.setValue([l, r])
        grn.basedur=np.interp(y, [0, 50], [0.001, 1]).item()

    else:
        pat.time = 1000000 # stutter

        cutoff.setValue((np.interp(x, [0, 100], [10, 8000])).item())
        srscale.setValue((np.interp(y, [0, 100], [0.01, 0.6])).item())

rec = OscDataReceive(port=9900, address="/face_data", function=getDataMessage)


s.gui(locals())