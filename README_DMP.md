# METEOR Batch Execution Guide

This repo previously used a script named `run_batch.sh`, but it is currently kept as `exec_batch.sh` so it is not excluded by the repo's gitignore rules.

The script wraps `python3 run_batch.py` and lets you run batch BEV autolabel jobs with simple defaults.

## File

- Script: `exec_batch.sh`
- Python entry: `run_batch.py`

## Prerequisites

Use the project environment:

```bash
conda activate meteor
cd /home/tatsuhiko/workspace/meteor/METEOR
```

## Basic usage

Run with the default settings:

```bash
./exec_batch.sh
```

This uses:

- scenes: `out/demo6/scenes.txt`
- output: `out/autolabel`
- root: `out/demo6`
- workers: `16`
- stride: `1`

## Custom usage

### 1) Use a different scene list

```bash
./exec_batch.sh data/highway_day/scenes.txt out/highway_autolabel
```

### 2) Use a custom dataset root

```bash
BEVLANE_ROOT=/home/tatsuhiko/workspace/meteor/METEOR/data/highway_day ./exec_batch.sh
```

### 3) Change worker count

`WORKERS` controls the number of CPU processes used by the autolabel step.
It is not a GPU count.

```bash
WORKERS=8 ./exec_batch.sh
```

### 4) Change stride

`STRIDE` controls how many frames are skipped between processed frames.

```bash
STRIDE=2 ./exec_batch.sh
```

### 5) Use both custom values together

```bash
BEVLANE_ROOT=out/demo6 WORKERS=8 STRIDE=2 ./exec_batch.sh out/demo6/scenes.txt out/autolabel_stride2
```

## What the script passes to Python

The shell script executes:

```bash
BEVLANE_ROOT="$ROOT" python3 run_batch.py \
  --scenes "$SCENES" \
  --out "$OUT" \
  --workers "$WORKERS" \
  --stride "$STRIDE"
```

So the script is simply a convenience wrapper around the following command:

```bash
python3 run_batch.py --scenes out/demo6/scenes.txt --out out/autolabel --workers 16 --stride 1
```

## Notes

- `--scenes` can be:
  - a file containing one scene name per line, or
  - a comma-separated list such as `sceneA,sceneB,sceneC`
- `--out` is the directory used for generated outputs.
- `WORKERS` is a CPU value; use it based on the machine's CPU core count and memory capacity.
- If the output already exists, the batch script will skip completed scenes in `run_batch.py`.

## Example workflow

```bash
conda activate meteor
cd /home/tatsuhiko/workspace/meteor/METEOR
chmod +x exec_batch.sh
./exec_batch.sh
```

This is the easiest way to trigger batch autolabel runs without typing the full Python arguments every time.
