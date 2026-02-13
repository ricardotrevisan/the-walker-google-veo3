import os
import time
import json
import subprocess
from pathlib import Path

from google import genai
from google.genai import errors
from google.genai import types
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
MODEL = "veo-3.1-fast-generate-preview"  # Veo 3.1 Fast Preview model id :contentReference[oaicite:4]{index=4}
OUT_DIR = Path("out_scenes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Polling
POLL_SECONDS = 10

# Video config
ASPECT_RATIO = "16:9"   # or "9:16" :contentReference[oaicite:5]{index=5}
RESOLUTION = "720p"     # 720p supports 4/6/8s; 1080p/4k only 8s :contentReference[oaicite:6]{index=6}
DURATION_SECONDS = 6    # 4, 6, or 8 :contentReference[oaicite:7]{index=7}

# Optional: steer away from unwanted artifacts
NEGATIVE_PROMPT = "cartoon, drawing, low quality, text overlays, watermarks, subtitles"

# Your shared “style bible” to repeat in every scene to keep consistency
STYLE_BIBLE = (
    "pastel 1960s aesthetic, cold ghostly atmosphere, surreal but calm and intense, "
    "soft diffused volumetric light, symmetrical cinematic framing, minimalist corridor"
)
SCENES_FILE = Path("storytelling.json")


def require_env():
    # Client will read GEMINI_API_KEY from environment in many setups;
    # keep it explicit for clarity.
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable before running.\n"
            "Example: export GEMINI_API_KEY='...'\n"
        )


def load_scenes(file_path: Path) -> list[tuple[str, str]]:
    if not file_path.exists():
        raise RuntimeError(
            f"Scenes file not found: {file_path}\n"
            "Create JSON with this format: [{\"name\": \"scene_01\", \"prompt\": \"...\"}]"
        )

    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or not raw:
        raise RuntimeError("The scenes file must contain a non-empty JSON list.")

    scenes: list[tuple[str, str]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid item at position {i}: expected a JSON object.")
        name = item.get("name")
        prompt = item.get("prompt")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"Invalid item {i}: missing or empty 'name' field.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"Invalid item {i}: missing or empty 'prompt' field.")
        scenes.append((name.strip(), prompt.strip()))
    return scenes


def generate_one_scene(client: genai.Client, name: str, prompt: str) -> Path:
    """
    Generates one video scene via Veo and saves as mp4.
    Uses polling pattern from the official docs. :contentReference[oaicite:8]{index=8}
    """
    config = types.GenerateVideosConfig(
        aspect_ratio=ASPECT_RATIO,
        resolution=RESOLUTION,
        # Note: docs show durationSeconds; Python SDK commonly uses duration_seconds.
        duration_seconds=DURATION_SECONDS,
        negative_prompt=NEGATIVE_PROMPT,
        number_of_videos=1,
    )

    try:
        operation = client.models.generate_videos(
            model=MODEL,
            prompt=prompt,
            config=config,
        )
    except errors.ClientError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Gemini API quota exceeded (429 RESOURCE_EXHAUSTED).\n"
                "Check plan/billing and limits at:\n"
                "- https://ai.dev/rate-limit\n"
                "- https://ai.google.dev/gemini-api/docs/rate-limits"
            ) from exc
        raise

    while not operation.done:
        print(f"[{name}] waiting ({POLL_SECONDS}s)...")
        time.sleep(POLL_SECONDS)
        try:
            operation = client.operations.get(operation)
        except errors.ClientError as exc:
            if exc.code == 429:
                raise RuntimeError(
                    "Gemini API quota exceeded during polling (429 RESOURCE_EXHAUSTED).\n"
                    "Check plan/billing and limits at:\n"
                    "- https://ai.dev/rate-limit\n"
                    "- https://ai.google.dev/gemini-api/docs/rate-limits"
                ) from exc
            raise

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    out_path = OUT_DIR / f"{name}.mp4"
    generated_video.video.save(str(out_path))
    print(f"[{name}] saved -> {out_path}")
    return out_path


def concat_videos_ffmpeg(video_paths: list[Path], out_path: Path) -> None:
    """
    Concatenate mp4 segments using ffmpeg concat demuxer (no re-encode if codecs compatible).
    Requires ffmpeg installed.
    """
    list_file = OUT_DIR / "concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in video_paths:
            # ffmpeg concat demuxer expects this format
            f.write(f"file '{p.as_posix()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Final video saved -> {out_path}")


def main():
    load_dotenv()
    require_env()
    scenes = load_scenes(SCENES_FILE)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    paths = []
    for name, prompt in scenes[:1]:
        try:
            paths.append(generate_one_scene(client, name, prompt))
        except RuntimeError as exc:
            print(f"[{name}] error: {exc}")
            break

    if not paths:
        print("No scene was generated. Exiting without concatenation.")
        return

    if len(paths) == 1:
        print(f"Only 1 scene generated: {paths[0]}. Skipping concatenation.")
        return

    final_path = Path("final.mp4")
    concat_videos_ffmpeg(paths, final_path)


if __name__ == "__main__":
    main()
