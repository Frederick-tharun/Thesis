| Regime | Optimizer | Rank | Validation NRMSE (x) | Validation NRMSE (all) | Best score |
|---|---|---:|---:|---:|---:|
| periodic_spiking | Random forest | 1 | 2.622e-05 | 1.825e-05 | 2.986e-05 |
|  | Gaussian process | 2 | 0.0001409 | 9.582e-05 | 0.0001399 |
|  | GBRT | 3 | 0.0003666 | 0.0002481 | 0.0003885 |
|  | Random search | 4 | 0.000378 | 0.0002595 | 0.0003944 |
| periodic_bursting | Random forest | 1 | 0.0001403 | 0.0001159 | 0.0001625 |
|  | GBRT | 2 | 0.0001779 | 0.000149 | 0.0002131 |
|  | Gaussian process | 3 | 0.0004246 | 0.0003769 | 0.0005909 |
|  | Random search | 4 | 0.0008342 | 0.0006908 | 0.001016 |
| chaotic_bursting | GBRT | 1 | 0.0005964 | 0.000444 | 0.0006412 |
|  | Gaussian process | 2 | 0.002563 | 0.001923 | 0.003074 |
|  | Random forest | 3 | 0.005307 | 0.003893 | 0.006167 |
|  | Random search | 4 | 0.009939 | 0.007334 | 0.01146 |
