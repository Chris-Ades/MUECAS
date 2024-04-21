from pyo import *

s = Server().boot()

rec = OscReceive(port=9000, address=["/volume", "/frequency", "/cutoff", "/tremolo", "/detune", "/q"])
rec.setValue("/volume", 0.5)
rec.setValue("/frequency", 220)
rec.setValue("/cutoff", 4000)
rec.setValue("/tremolo", 1)
rec.setValue("/detune", 0.5)
rec.setValue("/q", 1)

volume = SigTo(rec["/volume"], time=0.05, mul=1, add=0)
frequency = SigTo(rec["/frequency"], time=0.5, mul=1, add=0)
cutoff_frequency = SigTo(rec["/cutoff"], time=0.05, mul=1, add=0)
tremolo = SigTo(rec["/tremolo"], time=0.05, mul=1, add=0)
detune = SigTo(rec["/detune"], time=0.05, mul=1, add=0)
q = SigTo(rec["/q"], time=0.05, mul=1, add=0)

snd = SuperSaw(freq=frequency, detune=detune, bal=0.7, mul=volume)*Sine(tremolo, mul=volume)

low_pass = MoogLP(snd, freq=cutoff_frequency)

bp = Reson(Noise(mul=volume/2), freq=frequency/2, q=q)

tot = Mix([low_pass, bp], 2)

comp0 = Compress(tot, thresh=-20, ratio=4, risetime=0.005, falltime=0.10, knee=0.5, mul=2).out(0)
comp1 = Compress(tot, thresh=-20, ratio=4, risetime=0.005, falltime=0.10, knee=0.5, mul=2).out(1)

s.gui(locals())