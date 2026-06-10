# Final prediction and control comparison

| Regime | Optimizer | N_res | density_p | rho | leak | input_scale | ridge | washout | Pred_RMSE_x | Pred_NRMSE_x | Pred_RMSE_all | Pred_NRMSE_all | Control_method | Best_K | Control_target_RMSE_state | Control_target_RMSE_x | Spike_reduction_percent | Control_energy | Settling_time | Control_stable | Output_folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chaotic_bursting | bo_loaded | 715.000 | 0.073625 | 0.512552 | 0.147481 | 0.280867 | 3.28e-08 | 372.000 | 0.026143 | 0.052985 | 0.035910 | 0.035587 | pyragas | 0.050000 | 1.117932 | 0.634003 | 45.455 | 0.000366 | 37.760 | Yes | outputs/chaotic_bursting/pyragas |
