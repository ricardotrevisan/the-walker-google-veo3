import os
import time
import json
import subprocess
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors
from google.genai import types
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
MODEL = "veo-3.1-generate-preview"
OUT_DIR = Path("out_scenes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Polling
POLL_SECONDS = 10

# Video config
ASPECT_RATIO = "16:9"   # or "9:16" :contentReference[oaicite:5]{index=5}
RESOLUTION = "1080p"   # Prefer higher fidelity for quality-first passes.
DURATION_SECONDS = 8    # Longer duration improves motion continuity.

# Optional: set a fixed seed for reproducible prompt iteration.
SEED = None
AUTO_CHAIN_SCENES = True

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


def _optional_path(value: Any, field_name: str, item_idx: int) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid item {item_idx}: '{field_name}' must be a non-empty string path.")
    p = Path(value.strip())
    if not p.exists():
        raise RuntimeError(f"Invalid item {item_idx}: '{field_name}' path does not exist: {p}")
    if not p.is_file():
        raise RuntimeError(f"Invalid item {item_idx}: '{field_name}' must be a file: {p}")
    return p


def load_scenes(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise RuntimeError(
            f"Scenes file not found: {file_path}\n"
            "Create JSON with this format: [{\"name\": \"scene_01\", \"prompt\": \"...\"}]"
        )

    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or not raw:
        raise RuntimeError("The scenes file must contain a non-empty JSON list.")

    scenes: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid item at position {i}: expected a JSON object.")
        name = item.get("name")
        prompt = item.get("prompt")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"Invalid item {i}: missing or empty 'name' field.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"Invalid item {i}: missing or empty 'prompt' field.")
        reference_images_raw = item.get("reference_images", [])
        if reference_images_raw is None:
            reference_images_raw = []
        if not isinstance(reference_images_raw, list):
            raise RuntimeError(f"Invalid item {i}: 'reference_images' must be a list of file paths.")
        if len(reference_images_raw) > 3:
            raise RuntimeError(f"Invalid item {i}: Veo 3.1 supports at most 3 reference images.")
        reference_images: list[Path] = []
        for idx, ref in enumerate(reference_images_raw, start=1):
            ref_path = _optional_path(ref, f"reference_images[{idx}]", i)
            if ref_path is not None:
                reference_images.append(ref_path)

        scenes.append(
            {
                "name": name.strip(),
                "prompt": prompt.strip(),
                "first_frame_image": _optional_path(item.get("first_frame_image"), "first_frame_image", i),
                "last_frame_image": _optional_path(item.get("last_frame_image"), "last_frame_image", i),
                "reference_images": reference_images,
            }
        )
    return scenes


def _load_image_from_file(path: Path) -> types.Image:
    if not hasattr(types.Image, "from_file"):
        raise RuntimeError(
            "This google-genai version does not support types.Image.from_file(). "
            "Upgrade with: pip install -U google-genai"
        )
    return types.Image.from_file(str(path))


def generate_one_scene(client: genai.Client, scene: dict[str, Any]) -> Path:
    """
    Generates one video scene via Veo and saves as mp4.
    Uses polling pattern from the official docs. :contentReference[oaicite:8]{index=8}
    """
    name = scene["name"]
    prompt = scene["prompt"]
    first_frame_image: Path | None = scene.get("first_frame_image")
    last_frame_image: Path | None = scene.get("last_frame_image")
    reference_images_paths: list[Path] = scene.get("reference_images", [])

    config_kwargs: dict[str, Any] = dict(
        aspect_ratio=ASPECT_RATIO,
        resolution=RESOLUTION,
        # Note: docs show durationSeconds; Python SDK commonly uses duration_seconds.
        duration_seconds=DURATION_SECONDS,
        negative_prompt=NEGATIVE_PROMPT,
        number_of_videos=1,
        seed=SEED,
    )
    if last_frame_image:
        config_kwargs["last_frame"] = _load_image_from_file(last_frame_image)
    if reference_images_paths:
        config_kwargs["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=_load_image_from_file(ref_path),
                reference_type="asset",
            )
            for ref_path in reference_images_paths
        ]
    config = types.GenerateVideosConfig(**config_kwargs)

    try:
        request_kwargs: dict[str, Any] = dict(
            model=MODEL,
            prompt=prompt,
            config=config,
        )
        if first_frame_image:
            request_kwargs["image"] = _load_image_from_file(first_frame_image)
        operation = client.models.generate_videos(
            **request_kwargs,
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


def extract_last_frame(video_path: Path, out_path: Path) -> Path:
    """
    Extracts an approximate last frame from a generated video.
    This output can be fed as the next scene's first_frame_image.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def main():
    load_dotenv()
    require_env()
    scenes = load_scenes(SCENES_FILE)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    paths = []
    previous_last_frame: Path | None = None
    for scene in scenes[:1]:
        name = scene["name"]
        if AUTO_CHAIN_SCENES and previous_last_frame and not scene.get("first_frame_image"):
            scene["first_frame_image"] = previous_last_frame
            print(f"[{name}] auto-chain first frame <- {previous_last_frame}")
        try:
            video_path = generate_one_scene(client, scene)
            paths.append(video_path)
            if AUTO_CHAIN_SCENES:
                previous_last_frame = extract_last_frame(
                    video_path,
                    OUT_DIR / f"{name}_last_frame.png",
                )
        except RuntimeError as exc:
            print(f"[{name}] error: {exc}")
            break
        except subprocess.CalledProcessError:
            print(f"[{name}] warning: failed to extract last frame; continuing without auto-chain.")
            previous_last_frame = None

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
