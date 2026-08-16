# Chapter 2 Hindmarsh–Rose dynamics summary

These labels are preliminary diagnostics, not final scientific classifications.

| current_I | retained_samples | transient_steps | spike_count | mean_isi | isi_std | isi_cv | burst_structure | burst_count | mean_spikes_per_burst | std_spikes_per_burst | mean_within_burst_isi | within_burst_isi_cv | mean_interburst_interval | interburst_interval_cv | largest_lyapunov_exponent | lyapunov_convergence | lyapunov_classification | half_window_consistency | preliminary_regime | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.67 | 100000 | 100000 | 14 | 68.945385 | 51.649604 | 0.74913796 | bursting | 7 | 2 | 0 | 21.127143 | 0.00021382633 | 145.86167 | 2.5550099e-05 | 0.00019531964 | not_converged | uncertain | inconsistent | periodic bursting | adaptive log-ISI split gap=1.775, prominence=6415, threshold=51.3376; max normalized state-mean shift=0.2436; max relative state-std shift=0.07598; mean-ISI relative shift=0.04621; Lyapunov not_converged, final-checkpoint tolerance=0.0005 |
| 3.20 | 100000 | 100000 | 31 | 31.944333 | 22.702841 | 0.71070011 | bursting | 7 | 4.4285714 | 1.1780302 | 21.699167 | 0.49832979 | 138.56 | 0.2262036 | 0.011420291 | converged | positive | inconsistent | chaotic bursting | adaptive log-ISI split gap=0.2607, prominence=7.764, threshold=57.8827; max normalized state-mean shift=0.03844; max relative state-std shift=0.129; mean-ISI relative shift=0.0262; Lyapunov converged, final-checkpoint tolerance=0.002284 |
| 3.29 | 100000 | 100000 | 32 | 31.666129 | 21.820112 | 0.68906788 | bursting | 8 | 3.875 | 0.33071891 | 19.11087 | 0.29058528 | 125.97286 | 0.014502454 | 0.0006856195 | converged | near zero | consistent | uncertain | adaptive log-ISI split gap=0.8525, prominence=181.5, threshold=43.9238; ignored 1 singleton spike group(s); max normalized state-mean shift=0.003748; max relative state-std shift=0.00349; mean-ISI relative shift=0.001028; Lyapunov converged, final-checkpoint tolerance=0.0005 |
| 3.34 | 100000 | 100000 | 28 | 33.912593 | 14.336658 | 0.42275323 | bursting | 9 | 2.5555556 | 0.49690399 | 21.605 | 0.17350495 | 107.02875 | 0.40688742 | 0.01224151 | converged | positive | inconsistent | chaotic bursting | adaptive log-ISI split gap=0.1871, prominence=5.97, threshold=30.6906; ignored 5 singleton spike group(s); max normalized state-mean shift=0.2773; max relative state-std shift=0.1574; mean-ISI relative shift=0.03427; Lyapunov converged, final-checkpoint tolerance=0.002448 |
| 3.50 | 100000 | 100000 | 32 | 31.736452 | 0.18108185 | 0.0057058 | tonic | 0 | NaN | NaN | NaN | NaN | NaN | NaN | -0.00071203092 | not_converged | uncertain | consistent | periodic spiking | no clear two-timescale ISI separation; max normalized state-mean shift=0.0189; max relative state-std shift=0.007513; mean-ISI relative shift=0.0002311; Lyapunov not_converged, final-checkpoint tolerance=0.0005 |

## Methods

Spikes are x peaks with height >= 0.0, prominence >= 0.5, and minimum distance 20 steps (0.2 model-time units).

The largest adjacent gap in sorted log-ISI values defines the adaptive candidate split. It must be >= 0.15 log units, >= 4 times the median other positive gap, and leave at least 2 intervals on each side. Each accepted burst contains at least 2 spikes. A regular tonic train is not split; ambiguous structure remains uncertain.

Periodic bursting requires the CV of within-burst ISIs, inter-burst intervals, and spikes per burst to be no greater than 15%, 15%, and 15%, respectively. Overall ISI CV is not used to decide whether bursting is periodic.

The half-window consistency check compares state means, state standard deviations, and mean ISI using a 10% tolerance. 'consistent' only means that the two retained halves have similar measurements. 'inconsistent' can also result from chaotic fluctuations or incomplete burst cycles and does not by itself show that the initial transient was insufficient.

Lyapunov estimates use a 100000-step transient, 500000 evaluation steps, and tangent renormalization every 10 steps. Running estimates are retained at 100000, 200000, 300000, 400000, 500000. The estimate is converged when both consecutive changes among the last three checkpoints are within the larger of 0.0005 and 20% of the final estimate.
