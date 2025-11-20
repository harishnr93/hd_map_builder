# Visualization Assets

This directory can store generated figures (PNG/SVG) and screenshots for documentation:

- `architecture.png` – exported from `docs/architecture.md` Mermaid diagram.
- `ply_example.png` – screenshot from CloudCompare/Meshlab visualizing `logs/fused_map.ply`.
- `ros_stream.gif` – terminal capture of `scripts/stream_localization.py` publishing poses.

Use `mermaid-cli` or VSCode Mermaid preview to export diagrams:

```bash
npx @mermaid-js/mermaid-cli -i docs/architecture.md -o docs/media/architecture.png
```

Screenshots can be placed here and referenced from `README.md`.
