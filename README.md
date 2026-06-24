# Hindmarsh-Rose ESN Prediction and Control

This keeps your original structure, but fixes the HR pipeline to use the full state:

```text
[x, y, z] -> [x_next, y_next, z_next]
```

The final plot still shows `x`, because `x` is the neuron voltage-like signal.

## Run baseline full-state recursive ESN

```bash
python main.py --dataset hr --no-opt
```

## Run one optimizer

```bash
python main.py --dataset hr --optimizer gp
```

## Compare optimizers

```bash
python compare_optimizers.py --dataset hr
```

Outputs go to:

```text
outputs/
```

## Chaotic-bursting control

```bash
# Linear feedback
python main.py --dataset hr --hr-mode chaotic_bursting --no-opt \
  --control --controller linear_feedback --auto-control-k

# Finite-time feedback
python main.py --dataset hr --hr-mode chaotic_bursting --no-opt \
  --control --controller finite_time --finite-s 0.8 --auto-control-k

# Pyragas delayed feedback
python main.py --dataset hr --hr-mode chaotic_bursting --no-opt \
  --control --controller pyragas --pyragas-delay 320 \
  --pyragas-sign -1 --auto-control-k
```

Existing outputs are preserved unless `--clean-output` is explicitly supplied.

## Regression tests

```bash
python -m unittest discover -s tests -v
```
