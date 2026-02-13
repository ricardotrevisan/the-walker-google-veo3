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
```

## Storytelling Source (JSON)

`storytelling.json` must be a JSON list with objects in this format:

```json
[
  {
    "name": "scene_01",
    "prompt": "Visual scene description...\nAudio: audio description..."
  }
]
```

Required fields per item:
- `name` (string, used as the `.mp4` file name)
- `prompt` (string, full scene prompt)

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
