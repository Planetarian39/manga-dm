---
title: Application Snapshot
description: Scope, validation record, and release boundary for the graduate-application version of manga-dm.
---

# Application snapshot

This page defines the public graduate-application candidate for `manga-dm`.

| Field | Value |
|---|---|
| Researcher | Hongyi Xu |
| Affiliation | Department of Physics, University of Toronto |
| Package version | `0.1.0` |
| Snapshot date | 2026-07-17 |
| Source revision | Pending final application commit; working-tree base `859d5d5` |
| Status | Application candidate; research software remains Alpha |
| Official site | `https://planetarian39.github.io/manga-dm/` |
| Repository | `https://github.com/Planetarian39/manga-dm` |

## Included public evidence

- manuscript-aligned method documentation and explicit implementation differences;
- four allowlisted single-galaxy method demonstrations;
- pinned posterior artifacts with byte counts, SHA-256 digests, and provenance;
- architecture, CLI workflow, configuration, inputs, outputs, and known limitations;
- automated source, build, route, and artifact-integrity checks.

## Verification record

The application candidate is accepted only when the following commands pass from a clean dependency installation:

```powershell
npm run docs:check
npm run docs:build
manga --help
```

The GitHub Pages workflow additionally runs Python documentation-tool tests and direct HTTPS checks against representative pages, assets, and the exact byte size of the 11743-9102 posterior file.

## Release boundary

This snapshot documents methods, software, and allowlisted single-galaxy artifacts. It excludes manuscript source files, aggregate sample products, population findings, discussion, conclusions, and novelty claims. The public CLI is not labeled as a complete manuscript-reproduction profile until its configuration, likelihood route, provenance, and numerical regressions are versioned together.

A final Git tag or GitHub release may pin the deployment commit after application review. Until then, the deployed GitHub Pages revision and repository history are the authoritative version record.
