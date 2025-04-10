# Rayleigh-Benard Convection 2D

PHYS 6260 Term Project

## Dependencies

This code is based on Taichi with a Python frontend.

```bash
# python 3.9
pip install taichi -U
pip install matplotlib
```

## Execution

See `./scripts/run.ps1`

```bash
python ./exec/rbc2d.py
```

Generate the video (require ffmpeg):

```powershell
./scripts/gen_video.ps1 ./exps/rbc-c0 T
```