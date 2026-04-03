# Architecture Validation Notes

This project direction is fundamentally sound, but a few implementation details should be locked down early so the system you build matches the problem instead of the outline.

## What Is Correct

- Start with time-series only. That is the right MVP for validating whether latent-space prediction error carries pre-failure signal.
- Treat "normal data" curation as a first-class product capability. If the normal corpus is contaminated, the model baseline is wrong.
- Use per-machine thresholds and trend logic instead of global cutoffs.
- Add pattern matching only after anomaly scoring exists. It is the layer that makes alerts actionable.

## Corrections To Apply While Building

### 1. Preserve raw timestamps and raw signals

Do not irreversibly align or bucket every modality inside ingestion. Ingestion should validate, timestamp, and store raw events. Alignment belongs in the feature/model pipeline so you can change windowing later without re-ingesting the world.

Recommended rule:

- Ingestion: raw events + quality flags + source metadata
- Training/inference prep: resampling, interpolation, masking, and multimodal alignment

### 2. Separate JEPA inputs from engineered features

If the core model is a temporal JEPA, its primary input should be normalized raw sensor windows, plus masking and quality information. Heavy rolling stats and pairwise correlations are useful for:

- baseline models
- dashboards and explainability
- fallback rules

They should not be assumed to be the main JEPA input unless experiments prove otherwise.

### 3. Keep raw anomaly score separate from alert severity

The outlined inference loop produces a raw distance-like score such as latent L2 error. That value is not naturally bounded to `0-1`. If you want a `0-1` UI or alert score, calibrate it separately per machine or machine class.

Recommended rule:

- `score`: raw anomaly metric used for thresholds and backtesting
- `severity`: calibrated `0-1` signal for operators and downstream routing

### 4. Pattern matching must be filtered, not naive nearest-neighbor

Nearest-neighbor search across all latent states will mostly retrieve similar operating regimes, not useful pre-failure analogs.

Pattern matching should:

- index labeled pre-failure windows or trajectories
- filter by asset class, operating mode, and compatible sensor layout
- store time-to-failure metadata
- return evidence only when similarity is high and the comparison set is relevant

### 5. "Exactly-once" is not your real guarantee

Kafka/Redpanda can provide strong delivery semantics in narrow cases, but end-to-end industrial pipelines are usually designed for idempotent, effectively-once processing. Build deduplication and replay safety into consumers and storage keys.

### 6. Feast, Kubernetes, and multimodal infra are not MVP requirements

For the MVP, the minimal defensible stack is smaller than the outline:

- raw files in Parquet
- metadata and events in Postgres/Timescale
- training notebooks or scripts in PyTorch
- offline backtesting on known failures
- a thin inference path after the signal is proven

Add Kafka, Feast, Triton, Ray, and multimodal encoders after the anomaly signal is validated.

### 7. Validate against strong non-JEPA baselines

If VL-JEPA is the thesis, you still need credible baselines:

- naive forecasting residuals
- TCN/Transformer forecasting
- reconstruction autoencoder
- isolation forest or one-class methods on engineered features

If JEPA does not beat these on lead time and false-positive burden, it should not be the production default.

## MVP Success Criteria To Track

- event-level recall on known failures
- median lead time before failure
- alerts per machine-day
- false positives per week per line
- operator-confirmed precision
- score drift by machine over time

These metrics matter more than model elegance. Build toward them first.
