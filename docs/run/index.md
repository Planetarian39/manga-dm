# Run the pipeline

The public workflow has five operational transitions: select targets, fit single galaxies, merge posterior samples, fit or diagnose the population model, and generate figures. Use configuration deliberately—current defaults are not a manuscript-aligned profile.

## Quick start

Install the package from a checkout and verify the public interface before configuring data-backed runs:

```powershell
pip install -e .
manga --help
manga stage1 --help
```

These commands verify installation without starting a scientific fit. Continue to [Installation](/run/installation) for environment setup and [CLI workflow](/run/cli-workflow) for data-backed commands.

## Recommended reading order

1. [Install the environment](/run/installation).
2. Review the [CLI workflow](/run/cli-workflow).
3. Set paths and thresholds through [configuration](/run/configuration).
4. Learn the [input and output schemas](/run/inputs-and-outputs).
5. Compare commands with the [manuscript-aligned method](/methods/).

<MethodStatus status="paper">

The manuscript workflow uses the manuscript-aligned screening values, quality
equation, and prior-corrected posterior-sample Stage 2 likelihood documented in
Methods.

</MethodStatus>

<MethodStatus status="implementation">

The CLI is runnable, but its fallback azimuth, predictive-HDI, quality preset,
and Stage 2 default do not collectively define the manuscript-aligned
configuration. Do not label an output “manuscript analysis reproduced” until the separate
code-alignment work and numerical regression checks are complete.

</MethodStatus>
