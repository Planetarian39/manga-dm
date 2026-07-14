# Posterior downloads and provenance

These are complete per-galaxy posterior sample files for the four approved method demonstrations. They are not the merged Stage 2 product, and there is no aggregate download.

## File schema

Every file is a NetCDF/HDF5 dataset with 4,000 aligned draws:

| Field | Type and shape | Meaning |
|---|---|---|
| `log10_M200_samples` | float64 `(sample,)` | Base-10 log halo-mass draws |
| `log10_c_samples` | float64 `(sample,)` | Base-10 log concentration draws |
| `sample` | int32 `(sample,)` | Draw coordinate |
| `sample_count` | int32 scalar | Stored draw count |
| `plate_ifu` | global attribute | MaNGA object identifier |

## Files

<CaseDownload
  galaxy="11743-9102"
  href="/downloads/posteriors/11743-9102_nfw_param_cm200_samples.nc"
  size="88,196 bytes"
  sha="9e4b2153aded926a7e7ccc8e8b88135e2614f5c9516f3ba28d0d9be447e3cafb"
/>

<CaseDownload
  galaxy="8994-12701"
  href="/downloads/posteriors/8994-12701_nfw_param_cm200_samples.nc"
  size="88,196 bytes"
  sha="b81f5cb4814c2145cdd448c0af0a3c0c1e6e3e44c308f28ef7ff4c145d7307c0"
/>

<CaseDownload
  galaxy="7977-3704"
  href="/downloads/posteriors/7977-3704_nfw_param_cm200_samples.nc"
  size="88,196 bytes"
  sha="fcb340382a00510b0acc4e576f11147c47a6236a5cbf358c2ae38a20244d9196"
/>

<CaseDownload
  galaxy="9493-6101"
  href="/downloads/posteriors/9493-6101_nfw_param_cm200_samples.nc"
  size="88,196 bytes"
  sha="4ca636a911518513780bb2ebc8ce0a5a099a3d0341f04be0e46b3839c8faca23"
/>

## Provenance record

The machine-readable [case-study provenance manifest](/meta/case-study-provenance.json) records each source archive entry, public destination, byte count, and SHA-256 digest. A separate [descriptive summary file](/meta/case-study-summaries.json) contains only per-galaxy quantiles and correlation used on these pages.

To verify a file on Windows:

```powershell
Get-FileHash -Algorithm SHA256 .\11743-9102_nfw_param_cm200_samples.nc
```

Use these artifacts with the configuration and implementation-status context documented on this site. File availability alone is not a quality-pass certificate.
