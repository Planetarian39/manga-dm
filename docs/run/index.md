# Run the pipeline

The public workflow has five operational transitions: select targets, fit single galaxies, merge posterior samples, fit or diagnose the population model, and generate figures. Use configuration deliberately—current defaults are not a paper-aligned profile.

## Recommended reading order

1. [Install the environment](/run/installation).
2. Review the [CLI workflow](/run/cli-workflow).
3. Set paths and thresholds through [configuration](/run/configuration).
4. Learn the [input and output schemas](/run/inputs-and-outputs).
5. Compare commands with the [finalized-paper method](/methods/).

<MethodStatus status="paper">

The finalized workflow uses the paper-aligned screening values, quality
equation, and prior-corrected posterior-sample Stage 2 likelihood documented in
Methods.

</MethodStatus>

<MethodStatus status="implementation">

The CLI is runnable, but its fallback azimuth, predictive-HDI, quality preset,
and Stage 2 default do not collectively define the finalized-paper
configuration. Do not label an output “paper reproduced” until the separate
code-alignment work and numerical regression checks are complete.

</MethodStatus>
