# Demo media

The README's click-to-play block points at files in this directory:

| File | Purpose |
|---|---|
| `start-demo-poster.png` | Thumbnail shown in the README |
| `start-demo.mp4` | The recording itself (opens on click) |
| `start-demo.gif` | Inline-friendly alternative |
| `start-demo.cast` | asciinema source — re-renders the above |

Generate them all with:

```bash
brew install asciinema agg ffmpeg
./scripts/record_demo.sh
```

Until they exist the README block renders as a broken image with its alt text
intact, and the note beneath it tells visitors to run `python scripts/demo_flight.py`
instead. That is deliberate: a repository should never look finished when it is
not.

GitHub will not play a `<video>` tag inside a README, which is why the poster
image links to the MP4 rather than embedding it.
