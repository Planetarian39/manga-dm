# CLI workflow

The CLI parses global configuration first and dispatches each subcommand to the corresponding pipeline module. Commands below describe current public behavior; method pages separately document finalized-paper values.

## 1. Select targets

```powershell
manga select --download
```

Use `--ifu-file` to choose the output plate-IFU list. `--download` requests the data products needed by later stages.

## 2. Fit single galaxies

```powershell
manga stage1 --ifu test --nfw
manga stage1 --ifu 11743-9102 --nfw
manga stage1 --ifu all --nfw --n-cores 4
```

Stage 1 prepares the velocity map, fits the empirical rotation curve, applies screening, and optionally runs the NFW model. `test`, `all`, and a specific plate-IFU are accepted dispatch modes.

## 3. Merge posterior samples

```powershell
manga merge --ifu-file data/plateifus.txt
```

The merge step reads allowlisted per-IFU posterior files and creates the common sample product used by Stage 2.

## 4. Fit or diagnose the population model

```powershell
manga stage2 --fit --quality-cut recommended
manga stage2 --diagnose --quality-cut recommended
```

<MethodStatus status="paper">

The finalized Stage 2 method uses the prior-corrected posterior-sample
likelihood and the finalized quality gate.

</MethodStatus>

<MethodStatus status="implementation">

Current population code defaults to the GMM path unless explicitly configured
otherwise, and the public CLI has no complete paper-aligned Stage 2 switch.
Treat `recommended` as an implementation preset, not the finalized paper gate.

</MethodStatus>

## 5. Generate figures or robustness samples

```powershell
manga figures --ifu 8994-12701 7977-3704 --output-dir figures
manga sample --n 10
```

Figure generation is a local analysis capability. This public site publishes only allowlisted single-galaxy figures and does not distribute aggregate paper figures or robustness outcomes.

## Global overrides

All subcommands accept global options before the subcommand:

```powershell
manga --config config.toml --data-dir D:\manga-data --result-dir D:\manga-results stage1 --ifu test --nfw
```

Use [configuration](/run/configuration) for lookup and path rules.
