# Working Style & Communication

- Expects the assistant to implement the full richness of a described idea; complains when a previous "LLM implementation" flattened his spec ("simplified it way too much"). Preserve architectural intent rather than reducing it to the simplest workable version. Confidence: 0.6
- Expects systems to be genuinely end-to-end and to hold up on out-of-distribution inputs, not just on in-distribution/templated benchmarks; treat OOD behaviour as a real requirement, and prefers pointing to papers/primary sources over ad-hoc guesses. Expects new capabilities to be wired into the UI (e.g. benchmark tab), not left only in standalone scripts. Confidence: 0.7
- Wants project documentation (README) to explain the architecture step by step with a diagram, and to include measured metrics and method comparisons — not just prose describing the system. Confidence: 0.55
- Prefers additive changes over destructive ones: when new evaluation data or fixtures are introduced, keep the existing set intact ("keep the old dataset but make this new one") rather than overwriting it, so old and new results stay comparable. Confidence: 0.55
