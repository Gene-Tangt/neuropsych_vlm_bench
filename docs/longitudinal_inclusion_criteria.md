# Longitudinal analysis: inclusion criteria

## 1. Available via API call
- The model can be called through the provider's API at the time of the study (models that are retired or not offered to us are out).
- It is a general-purpose chat model that accepts images. Specialised variants (audio, realtime, search, transcription, speech, image generation, coding-only, instruct/completion-only, open-weight releases, rolling "latest" aliases, premium "pro" tiers) are left out.
- Where a provider offers both a fixed, dated version and a moving alias of the same model, the fixed version is used.
- The model must answer one real benchmark trial with images before it is run.
- The model must complete all tests in the benchmark to be included in the analysis.

## 2. Can reasoning be turned off?
- Models without reasoning: run as they are.
- Reasoning models: run with reasoning turned off through the provider's control for it (e.g. effort set to "none", or thinking disabled).
- Reasoning only counts as off if the response uses 0 reasoning tokens.

## 2.5 If reasoning cannot be turned off
- Run at the lowest reasoning setting the model accepts.
- If no reasoning setting is accepted, run at the provider's default.
- The model is still included but flagged, and flagged models are marked in plots and tables.

## 3. Token limits
- Models without reasoning: 1,000 max output tokens.
- Flagged models: 4,000 max output tokens, because reasoning tokens count toward the limit.
- Temperature and other sampling settings are left at the provider's default.
- Truncated responses (cut off at the token limit) are counted and reported.
