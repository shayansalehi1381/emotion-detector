# Demo Video Recording Guide

A short video (30-60 seconds) of the live app is enough and effective for showing in a
resume/interview or attaching to the project. It doesn't need to be long.

## Before Recording

1. Make sure `checkpoints/best.pt` exists (Phase 3's output).
2. Set up good lighting — Haar Cascade finds faces later or more weakly in low light.
3. Run the app once to confirm FPS and face detection work correctly:
   ```bash
   PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/live_demo.py
   ```

## Screen Recording Tools (Windows)

- Simplest option: **Xbox Game Bar** (`Win + G`) — activate it on the app window, hit
  the record button. Output file is saved to `Videos/Captures`.
- Alternative: OBS Studio (free) for more control over output quality and framerate.

## What to Show (Suggested Scenario)

1. Genuinely show 3-4 different expressions (e.g. smile → Happy, frown → Angry,
   neutral face → Neutral, look surprised → Surprise). Since Happy and Surprise have
   higher accuracy (per `docs/PHASE_4_REPORT.md`), make sure to show these two so the
   model's best performance is visible.
2. Move your head slowly to show the face-detection box tracking.
3. Wait a few seconds between changing expressions so the moving average (smoothing)
   can adapt to the new expression (about 8 frames, less than a second at normal FPS).
4. At the end, press the `q` key so the app's clean exit is also visible in the video.

## After Recording

- Upload the video to `results/demo.mp4` (optional, it's in `.gitignore` since video
  files can be large) or directly to whatever platform you're presenting your
  resume/interview on.
- If you also want to make a short GIF (to show in a GitHub README), you can convert it
  with a tool like ffmpeg:
  ```bash
  ffmpeg -i results/demo.mp4 -vf "fps=10,scale=480:-1" results/demo.gif
  ```
