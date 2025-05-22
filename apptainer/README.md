# [Apptainer](https://github.com/apptainer/apptainer/releases/tag/v1.4.1)(Singularity) for Segment-Anything tools

## How to build

```bash
apptainer build --fakeroot --sandbox sam2_langsplat_sandbox sam2_cu124.def
```

## How to run

```bash
apptainer shell --nv sam2_langsplat_sandbox
source /entrypoint.sh
```

## How to install/uninstall other packages in sandbox

```bash
apptainer shell --fakeroot --writable --nv sam2_langsplat_sandbox
source /entrypoint.sh

apt install ...
pip install ...
```
