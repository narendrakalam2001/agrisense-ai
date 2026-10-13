import h5py
import json

MODEL_PATH = "models/efficientnet_plantvillage.h5"

with h5py.File(MODEL_PATH, "r+") as f:
    raw = f.attrs["model_config"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    config = json.loads(raw)

    def strip_quant(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for v in obj.values():
                strip_quant(v)
        elif isinstance(obj, list):
            for item in obj:
                strip_quant(item)

    strip_quant(config)
    f.attrs["model_config"] = json.dumps(config)

print("Patched successfully — quantization_config removed from all layer configs.")