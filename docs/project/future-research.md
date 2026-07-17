# Future research

The current pipeline uses an empirical rotation curve as a screening and initialization layer before the physical mass model. A future extension could study the shape of the inner rise as an explicit, uncertainty-aware descriptor.

## A possible shape representation

A compact research representation might combine the turnover scale, inner gradient, outer slope, and posterior uncertainty rather than assigning a visual class to one best-fit curve. That representation would need to remain separate from the current manuscript method until its selection effects and numerical behavior are validated.

## Questions to resolve

- Is a shape descriptor stable to beam smearing, inclination uncertainty, and radial coverage?
- Can it be estimated without reusing the same data twice in later inference?
- Which posterior-derived quantities remain identifiable for sparsely sampled inner radii?
- How should selection and measurement error propagate into any population study?

## Status

This is a research direction, not a current pipeline promise or a published result. It is intentionally isolated from the method and run guides.
