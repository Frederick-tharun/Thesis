# Corrected Hindmarsh-Rose ESN Project

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
