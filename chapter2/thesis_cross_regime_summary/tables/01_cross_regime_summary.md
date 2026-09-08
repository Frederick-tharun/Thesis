# Cross-regime summary

All metrics are aggregated over the saved fixed-short final-evaluation rollouts. IQR is Q3 − Q1.

| Direction | Train regime | Test regime | Representative current(s) or tested currents | Median NRMSE | IQR NRMSE | Median VPT | Divergence rate | Numerical failure rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RR | Regular | Regular | Tested: 1.67; 3.29; 3.50; representative: 3.50 | 0.00373482 | 0.0113256 | 80.00 | 0.0% | 0.0% |
| RC | Regular | Chaotic | Tested: 3.20; 3.34; representative: 3.20 | 1.02537 | 0.449378 | 12.50 | 0.0% | 0.0% |
| CC | Chaotic | Chaotic | Tested: 3.20; 3.34; representative: 3.20 | 0.00388366 | 0.00501994 | 80.00 | 0.0% | 0.0% |
| CR | Chaotic | Regular | Tested: 1.67; 3.29; 3.50; representative: 1.67 | 5.38751 | 13.25 | 0.09 | 80.0% | 0.0% |
