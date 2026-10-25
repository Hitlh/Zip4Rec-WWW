# Selected CD Layout

[compact.py](compact.py) contains the selected CD compact layout. [selection.json](selection.json) records its selection.

Pages use top/bottom sections, a width of 640 pixels, DejaVuSans 14-pixel body text, a 16-pixel line advance, and height rounded up to a multiple of 32 pixels. The code renders user context and one candidate item into an image.

Run from the repository root:

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout optical_compression/selected \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

The input is module 1's `datasets/`. Outputs include images, multimodal records, and actual token statistics for all three splits. The `render_compact(task)` interface accepts `sample_id`, `profile` (user text), `item` (candidate fields), and `path` (output PNG). Item fields are `title`, `artists`, `genres`, `era`, `moods`, and `summary`; artists, genres, and moods are lists.

For a new dataset, follow the search workflow in [module 2](../README.md), then point `--layout` to its exported `frozen/` directory.
