# Final prediction and control comparison

| Regime | Optimizer | N_res | density_p | rho | leak | input_scale | ridge | washout | Pred_RMSE_x | Pred_NRMSE_x | Pred_RMSE_all | Pred_NRMSE_all | Control_method | Best_K | Control_target_RMSE_state | Control_target_RMSE_x | Spike_reduction_percent | Control_energy | Settling_time | Control_stable | Output_folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<<<<<<< HEAD
| periodic_spiking | gp | 250.000 | 0.200000 | 0.450000 | 0.247776 | 0.440754 | 1.00e-08 | 500.000 | 0.000852 | 0.001642 | 0.001209 | 0.001006 | Linear feedback | 0.983214 | 2.73e-05 | 5.70e-06 | 100.000 | 7.65e-06 | 0.000000 | Yes | outputs\periodic_spiking |
| periodic_bursting | forest | 679.000 | 0.114799 | 0.487485 | 0.480519 | 0.336363 | 1.39e-08 | 153.000 | 0.000299 | 0.000503 | 0.000421 | 0.000343 | Linear feedback | 0.983214 | 5.56e-05 | 1.23e-05 | 100.000 | 3.18e-05 | 0.000000 | Yes | outputs\periodic_bursting |
| chaotic_bursting | gbrt | 419.000 | 0.166094 | 0.571785 | 0.329853 | 0.194551 | 4.89e-08 | 388.000 | 0.013312 | 0.026980 | 0.018496 | 0.018186 | Linear feedback | 0.983214 | 2.03e-05 | 5.63e-06 | 100.000 | 4.23e-06 | 0.000000 | Yes | outputs\chaotic_bursting |
=======
| chaotic_bursting | dummy | 715.000 | 0.073625 | 0.512552 | 0.147481 | 0.280867 | 3.28e-08 | 372.000 | 0.026533 | 0.053775 | 0.036443 | 0.036118 |  |  |  |  |  |  |  |  | outputs/chaotic_bursting |
>>>>>>> d87eeb28 (Adde linear feedback control)
