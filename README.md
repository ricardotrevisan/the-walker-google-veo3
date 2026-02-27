# Google Veo 3 Storytelling (Gemini API)

Python script to generate video scenes with Veo via Gemini API using `storytelling.json` as input.

## Requirements

- Python 3.10+
- `ffmpeg` installed on your system (optional; required only to concatenate multiple scenes)
- Gemini API key with access to the video endpoint

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create/edit the `.env` file:

```env
GEMINI_API_KEY="YOUR_KEY_HERE"
GEMINI_MODEL="veo-3.1-generate-preview"
```

## Storytelling Source (JSON)

`storytelling.json` must be a JSON list with objects in this format:

```json
[
  {
    "name": "scene_01",
    "prompt": "Visual scene description...\nAudio: audio description...",
    "duration_seconds": 4,
    "first_frame_image": "assets/scene_01_start.png",
    "last_frame_image": "assets/scene_01_end.png",
    "reference_images": [
      "assets/character_ref_front.png",
      "assets/character_ref_profile.png"
    ]
  }
]
```

Required fields per item:
- `name` (string, used as the `.mp4` file name)
- `prompt` (string, full scene prompt)

Optional fields per item:
- `duration_seconds` (integer >= 4): clip duration for that scene; if omitted, falls back to `.env`/default `DURATION_SECONDS` (also >= 4)
- `first_frame_image` (string path): uses this image as the starting frame/conditioning image
- `last_frame_image` (string path): asks Veo to end the clip near this target frame
- `reference_images` (list of up to 3 string paths): identity/style anchors for continuity (useful for same character)

## Run

```bash
python3 storytelling.py
```

Outputs:
- Individual scenes in `out_scenes/`
- Final concatenated video in `final.mp4` (only when 2+ scenes are processed)

## Current Behavior

- The script is currently configured to process only the first scene (`scenes[:1]`).
- If no scene is generated, the script exits without concatenation.
- If only 1 scene is generated, it keeps just that scene file.

## Continuity Tips (Same Character)

- Reuse the same `reference_images` for each scene.
- For scene N+1, set `first_frame_image` to a still frame exported from scene N.
- Keep core character description and wardrobe wording consistent in every `prompt`.
- Use a fixed `SEED` in `storytelling.py` while iterating prompts for stable results.
- The script can auto-chain frames across scenes (`AUTO_CHAIN_SCENES = True`): it extracts `out_scenes/<scene>_last_frame.png` and uses it as the next scene start frame when `first_frame_image` is not provided.

## Common Errors

### 429 RESOURCE_EXHAUSTED

This means quota/limit exceeded for your API key/project.

Check:
- https://ai.dev/rate-limit
- https://ai.google.dev/gemini-api/docs/rate-limits

## Main Files

- `storytelling.py`: video generation, polling, error handling, and concatenation
- `storytelling.json`: scenes source
- `.env`: local credentials
- `requirements.txt`: Python dependencies
