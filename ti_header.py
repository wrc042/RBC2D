import taichi as ti
import numpy as np

ti_device = "cuda"
if ti_device == "cuda":
    ti_ti_device = ti.cuda
ti_debug = False
ti_fp = ti.f32
ti_cache = True
ti_cache_dir = "__ticache__"

ti.init(
    arch=ti_ti_device,
    debug=ti_debug,
    default_fp=ti_fp,
    offline_cache=ti_cache,
    offline_cache_file_path=ti_cache_dir,
    random_seed=42,
)

np.random.seed(42)