# Enhancements TODO

- [ ] **Model lifecycle**: Keep analyzer/planner models loaded across chunks, add optional worker pool, and avoid redundant device cache clears to reduce per-chunk warm-up time without impacting inference quality.
- [ ] **Chunk preprocessing cache**: Reuse ffprobe metadata and mono audio extracts across retries/reruns, ideally captured during splitting, to eliminate duplicate `ffmpeg` work while preserving consistent analysis inputs.
- [ ] **Parallel preparation**: Overlap scene detection, transcription, and shot context building on background threads while the model is inferencing, ensuring only one device-bound forward pass at a time to maintain stability.
- [x] **FFmpeg acceleration options**: Detect available hardware encoders (videotoolbox/cuda) and expose a flag to use them for trims while keeping output quality controls (`crf`, bitrate) unchanged.
- [ ] **Audio mixing efficiency**: Cache decoded music stems and streamline `_mix_music` when cues align with the clip, potentially incorporating `sidechaincompress` for consistent loudness in a single filter graph.
- [x] **Resilience improvements**: Add retries and pre-flight checks around external commands, persist intermediate JSON to allow resumable runs, and surface missing assets (e.g., music tracks) before rendering starts.
- [ ] **Grounded video comprehension**: Replace chunk-level inference with a per-shot frame analyzer that samples multiple frames per detected shot, feeds them alongside transcripts/audio cues, and enforces deterministic decoding (low temperature, no sampling) to curb hallucinations.
- [ ] **Prompt & schema tightening**: Rewrite analyzer prompts to forbid invented entities, require empty lists when uncertain, and align language with stricter JSON schema validation; add automated schema/type checks and timestamp bound enforcement before accepting model output.
- [ ] **Stricter validation tooling**: Extend `validators.py` to assert timestamps fall within chunk duration, field types match the schema, confidences exceed configurable thresholds, and reject patterned/low-variance timelines indicative of fabricated content.
