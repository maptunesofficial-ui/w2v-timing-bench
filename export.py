"""CI step: export public wav2vec2 CTC models (Apache-2.0) to fixed-shape ONNX (6 s of 16 kHz audio) + a reference clip and the PyTorch reference ids.
Open-source models and a public-domain clip only."""
import json, sys, wave
import numpy as np, torch
from transformers import AutoModelForCTC

N = 96000  # 6 s
REPOS = {"base": "facebook/wav2vec2-base-960h", "large": "facebook/wav2vec2-large-960h-lv60-self"}
w = wave.open("jfk.wav"); assert w.getframerate() == 16000 and w.getnchannels() == 1
pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
x = pcm[:N]; x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
x.astype(np.float32).tofile("Resources/clip.f32")
ref = {}
for name, repo in REPOS.items():
    m = AutoModelForCTC.from_pretrained(repo).eval()
    t = torch.from_numpy(x)[None]
    with torch.no_grad():
        lg = m(t).logits
    ref[name] = {"ids": lg[0].argmax(-1).tolist(), "T": lg.shape[1], "V": lg.shape[2]}
    torch.onnx.export(m, (t,), f"Resources/w2v-{name}.onnx", input_names=["input_values"], output_names=["logits"], opset_version=17, dynamo=False)
    print(name, "exported", lg.shape, flush=True)
    del m
json.dump(ref, open("Resources/ref.json", "w"))
json.dump(json.load(open("vocab_base.json")), open("Resources/vocab.json", "w"))
