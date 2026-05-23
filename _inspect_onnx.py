import onnx
import numpy as np
from onnx import numpy_helper

p = "unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45_tag/model_38500_all_6_tag.onnx"
m = onnx.load(p)
g = m.graph

inits = {init.name: numpy_helper.to_array(init) for init in g.initializer}

mean = inits.get("policy.obs_normalizer._mean")
div = inits.get("onnx::Div_47")
print(f"_mean shape: {mean.shape}, _div shape: {div.shape}")
print()
print("idx |   mean    |    div     | abs(mean) |")
print("----|-----------|------------|-----------|")
for i in range(mean.size):
    val = mean.reshape(-1)[i]
    sval = div.reshape(-1)[i]
    print(f"{i:3d} | {val:+.4f}  | {sval:+.4f}   | {abs(val):.4f}")
