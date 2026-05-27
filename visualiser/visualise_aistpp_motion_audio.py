#!/usr/bin/env python3
"""Render one AIST++/DMD motion-conditioning sample beside its audio.

The output is a single GIF by default. It intentionally writes only the
visualization artifact to the output directory.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "dmd_aistpp_legacy_2026-05-26"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "aistpp_visualisations"

SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

SMPL_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.05858135, -0.08228004, -0.01766408],
        [-0.06030973, -0.09051332, -0.01354254],
        [0.00443945, 0.12440352, -0.03838522],
        [0.04345142, -0.38646945, 0.008037],
        [-0.04325663, -0.38368791, -0.00484304],
        [0.00448844, 0.1379564, 0.02682033],
        [-0.01479032, -0.42687458, -0.037428],
        [0.01905555, -0.4200455, -0.03456167],
        [-0.00226458, 0.05603239, 0.00285505],
        [0.04105436, -0.06028581, 0.12204243],
        [-0.03483987, -0.06210566, 0.13032329],
        [-0.0133902, 0.21163553, -0.03346758],
        [0.07170245, 0.11399969, -0.01889817],
        [-0.08295366, 0.11247234, -0.02370739],
        [0.01011321, 0.08893734, 0.05040987],
        [0.12292141, 0.04520509, -0.019046],
        [-0.11322832, 0.04685326, -0.00847207],
        [0.2553319, -0.01564902, -0.02294649],
        [-0.26012748, -0.01436928, -0.03126873],
        [0.26570925, 0.01269811, -0.00737473],
        [-0.26910836, 0.00679372, -0.00602676],
        [0.08669055, -0.01063603, -0.01559429],
        [-0.0887537, -0.00865157, -0.01010708],
    ],
    dtype=np.float32,
)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def load_encodings(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        encodings = pickle.load(handle)
    if not isinstance(encodings, dict):
        raise ValueError(f"Expected a dict in {path}")
    return encodings


def resolve_key(encodings: dict[str, np.ndarray], audio_dirs: list[Path], key: str | None) -> str:
    keys = [key] if key else sorted(encodings)
    for candidate in keys:
        if candidate not in encodings:
            continue
        if any((audio_dir / f"{candidate}.wav").exists() for audio_dir in audio_dirs):
            return candidate
    if key:
        raise FileNotFoundError(f"Key {key!r} was not found with a matching wav file")
    raise FileNotFoundError("No encoding key has a matching wav file")


def resolve_audio(audio_dirs: list[Path], key: str) -> Path:
    for audio_dir in audio_dirs:
        candidate = audio_dir / f"{key}.wav"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No wav found for {key}")


def extract_dmd_joint_positions(encoding: np.ndarray) -> np.ndarray:
    if encoding.ndim != 2 or encoding.shape[0] < 1:
        raise ValueError(f"Expected [frames, dim] encoding, got {encoding.shape}")
    motion_dim = encoding.shape[1] - 10 if encoding.shape[1] >= 370 else encoding.shape[1]
    if motion_dim < 360:
        raise ValueError(
            f"Need at least 360 DMD motion channels to draw joint positions; got {encoding.shape[1]}"
        )
    motion = encoding[:, :360].reshape(encoding.shape[0], 24, 15)
    return motion[:, :, 6:9].astype(np.float32)


def split_slice_key(key: str) -> tuple[str, int]:
    marker = "_slice"
    if marker not in key:
        return key, 0
    base, slice_text = key.rsplit(marker, 1)
    try:
        return base, int(slice_text)
    except ValueError:
        return key, 0


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = axis_angle.astype(np.float32)
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, angle, out=np.zeros_like(axis_angle), where=angle > 1e-8)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    c = np.cos(angle[..., 0])
    s = np.sin(angle[..., 0])
    one_c = 1.0 - c

    matrix = np.empty(axis_angle.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = c + x * x * one_c
    matrix[..., 0, 1] = x * y * one_c - z * s
    matrix[..., 0, 2] = x * z * one_c + y * s
    matrix[..., 1, 0] = y * x * one_c + z * s
    matrix[..., 1, 1] = c + y * y * one_c
    matrix[..., 1, 2] = y * z * one_c - x * s
    matrix[..., 2, 0] = z * x * one_c - y * s
    matrix[..., 2, 1] = z * y * one_c + x * s
    matrix[..., 2, 2] = c + z * z * one_c

    identity = np.eye(3, dtype=np.float32)
    matrix = np.where((angle <= 1e-8)[..., None], identity, matrix)
    return matrix


def forward_kinematics(rotations: np.ndarray, root_positions: np.ndarray) -> np.ndarray:
    local_rot = axis_angle_to_matrix(rotations)
    frames, joints = rotations.shape[:2]
    positions = np.zeros((frames, joints, 3), dtype=np.float32)
    global_rot = np.zeros((frames, joints, 3, 3), dtype=np.float32)

    for joint, parent in enumerate(SMPL_PARENTS):
        if parent == -1:
            positions[:, joint] = root_positions
            global_rot[:, joint] = local_rot[:, joint]
            continue
        offset = np.einsum("tij,j->ti", global_rot[:, parent], SMPL_OFFSETS[joint])
        positions[:, joint] = positions[:, parent] + offset
        global_rot[:, joint] = np.einsum("tij,tjk->tik", global_rot[:, parent], local_rot[:, joint])
    return positions


def load_raw_aist_positions(
    key: str,
    data_root: Path,
    target_frames: int,
    data_fps: int,
    raw_fps: int = 60,
) -> np.ndarray | None:
    base_key, slice_index = split_slice_key(key)
    motion_path = data_root / "aistplusplus_raw" / "motions" / f"{base_key}.pkl"
    if not motion_path.exists():
        return None

    with motion_path.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = pickle.load(handle)
    poses = np.asarray(payload["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(payload["smpl_trans"], dtype=np.float32)

    raw_step = max(1, raw_fps // data_fps)
    raw_start = slice_index * raw_fps * 5
    raw_end = raw_start + target_frames * raw_step
    if raw_end > len(poses):
        return None

    rotations = poses[raw_start:raw_end:raw_step]
    root_positions = trans[raw_start:raw_end:raw_step]
    if len(rotations) < target_frames:
        return None
    return forward_kinematics(rotations[:target_frames], root_positions[:target_frames])


def load_positions_for_key(
    key: str,
    encoding: np.ndarray,
    data_root: Path,
    data_fps: int,
    prefer_raw: bool,
) -> tuple[np.ndarray, str, str]:
    target_frames = int(encoding.shape[0])
    if prefer_raw:
        raw_positions = load_raw_aist_positions(key, data_root, target_frames, data_fps)
        if raw_positions is not None:
            return raw_positions, "raw", "AIST++ raw SMPL motion"
    return extract_dmd_joint_positions(encoding), "conditioning", "AIST++ DMD motion conditioning"


def make_waveform_points(audio: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.zeros(width), np.zeros(width)
    chunks = np.array_split(audio, width)
    lo = np.array([chunk.min() if chunk.size else 0.0 for chunk in chunks])
    hi = np.array([chunk.max() if chunk.size else 0.0 for chunk in chunks])
    return lo, hi


def make_spectrogram_image(audio: np.ndarray, sample_rate: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if audio.size == 0:
        return Image.new("RGB", size, (10, 13, 19))

    n_fft = 1024
    hop = 512
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    window = np.hanning(n_fft).astype(np.float32)
    frame_count = 1 + max(0, (audio.size - n_fft) // hop)
    spec = np.empty((n_fft // 2 + 1, frame_count), dtype=np.float32)
    for i in range(frame_count):
        start = i * hop
        frame = audio[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spec[:, i] = np.abs(np.fft.rfft(frame * window))

    spec = np.log1p(spec)
    spec = spec[: min(256, spec.shape[0])]
    spec -= spec.min()
    denom = spec.max()
    if denom > 0:
        spec /= denom
    spec = spec[::-1]

    gray = Image.fromarray(np.uint8(spec * 255), mode="L").resize(size, Image.Resampling.BILINEAR)
    values = np.asarray(gray).astype(np.float32) / 255.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(22 + 232 * np.clip(values * 1.4 - 0.35, 0, 1))
    rgb[..., 1] = np.uint8(35 + 210 * np.clip(values * 1.2, 0, 1))
    rgb[..., 2] = np.uint8(48 + 180 * np.clip(1.1 - values, 0, 1))
    return Image.fromarray(rgb, mode="RGB")


def project_pose(
    positions: np.ndarray,
    frame_index: int,
    rect: tuple[int, int, int, int],
    global_scale: float,
    source: str,
) -> np.ndarray:
    pose = positions[frame_index].copy()
    pose = pose - pose[0:1]

    if source == "raw":
        # Raw AIST++ SMPL poses are y-up.
        x = pose[:, 0]
        y = pose[:, 1]
        depth = pose[:, 2]
    else:
        # DMD preprocessing rotates AIST++ to z-up.
        x = pose[:, 0]
        y = pose[:, 2]
        depth = pose[:, 1]
    x = x + 0.18 * depth
    y = y + 0.08 * depth

    left, top, right, bottom = rect
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.58
    scale = min(right - left, bottom - top) * 0.42 / max(global_scale, 1e-6)
    screen = np.stack([cx + x * scale, cy - y * scale], axis=1)
    return screen


def compute_global_scale(positions: np.ndarray) -> float:
    centered = positions - positions[:, :1]
    y_axis = 1 if np.ptp(centered[:, :, 1]) > np.ptp(centered[:, :, 2]) else 2
    projected = centered[:, :, [0, y_axis]]
    spread = np.percentile(np.abs(projected), 99)
    return float(max(spread, 0.05))


def draw_panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=14, fill=(13, 17, 24), outline=(52, 60, 76), width=2)
    draw.text((left + 18, top + 14), title, fill=(232, 237, 245), font=font)
    draw.line((left + 16, top + 48, right - 16, top + 48), fill=(39, 48, 63), width=1)


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    positions_2d: np.ndarray,
    rect: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = rect
    floor_y = bottom - 48
    draw.line((left + 44, floor_y, right - 44, floor_y), fill=(45, 55, 68), width=2)

    for joint, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        x1, y1 = positions_2d[parent]
        x2, y2 = positions_2d[joint]
        color = (103, 211, 255)
        if joint in {1, 4, 7, 10, 16, 18, 20, 22}:
            color = (255, 187, 92)
        elif joint in {2, 5, 8, 11, 17, 19, 21, 23}:
            color = (117, 222, 147)
        draw.line((x1, y1, x2, y2), fill=color, width=5)

    for idx, (x, y) in enumerate(positions_2d):
        radius = 5 if idx != 0 else 7
        fill = (241, 246, 255) if idx != 0 else (255, 95, 115)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_waveform(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    lo: np.ndarray,
    hi: np.ndarray,
    progress: float,
) -> None:
    left, top, right, bottom = rect
    mid = (top + bottom) * 0.5
    amp = (bottom - top) * 0.42
    draw.rectangle(rect, fill=(8, 12, 19), outline=(43, 52, 67))
    draw.line((left, mid, right, mid), fill=(45, 55, 70), width=1)
    width = max(1, right - left)
    for x in range(width):
        idx = min(x, len(lo) - 1)
        y1 = mid - hi[idx] * amp
        y2 = mid - lo[idx] * amp
        draw.line((left + x, y1, left + x, y2), fill=(106, 209, 255), width=1)
    cursor_x = int(left + progress * width)
    draw.line((cursor_x, top, cursor_x, bottom), fill=(255, 88, 105), width=3)


def draw_progress_on_spectrogram(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    spectrogram: Image.Image,
    progress: float,
) -> None:
    left, top, right, bottom = rect
    image.paste(spectrogram, (left, top))
    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, outline=(43, 52, 67), width=1)
    cursor_x = int(left + progress * max(1, right - left))
    draw.line((cursor_x, top, cursor_x, bottom), fill=(255, 88, 105), width=3)


def render_gif(
    key: str,
    encoding: np.ndarray,
    audio_path: Path,
    data_root: Path,
    output_path: Path,
    fps: int,
    data_fps: int,
    size: tuple[int, int],
    prefer_raw: bool,
) -> None:
    positions, position_source, motion_title = load_positions_for_key(
        key=key,
        encoding=encoding,
        data_root=data_root,
        data_fps=data_fps,
        prefer_raw=prefer_raw,
    )
    audio, sample_rate = read_wav_mono(audio_path)
    duration = min(len(positions) / data_fps, audio.size / sample_rate if sample_rate else len(positions) / data_fps)
    total_data_frames = max(1, min(len(positions), int(round(duration * data_fps))))

    width, height = size
    title_font = load_font(22)
    small_font = load_font(16)
    tiny_font = load_font(13)

    margin = 24
    gap = 20
    skeleton_rect = (margin, margin, int(width * 0.55) - gap // 2, height - margin)
    audio_rect = (int(width * 0.55) + gap // 2, margin, width - margin, height - margin)
    waveform_rect = (
        audio_rect[0] + 18,
        audio_rect[1] + 68,
        audio_rect[2] - 18,
        audio_rect[1] + 210,
    )
    spectro_rect = (
        audio_rect[0] + 18,
        audio_rect[1] + 258,
        audio_rect[2] - 18,
        audio_rect[3] - 54,
    )

    waveform_lo, waveform_hi = make_waveform_points(audio, max(1, waveform_rect[2] - waveform_rect[0]))
    spectrogram = make_spectrogram_image(
        audio,
        sample_rate,
        (max(1, spectro_rect[2] - spectro_rect[0]), max(1, spectro_rect[3] - spectro_rect[1])),
    )
    global_scale = compute_global_scale(positions[:total_data_frames])

    step = max(1, round(data_fps / fps))
    frame_indices = list(range(0, total_data_frames, step))
    if frame_indices[-1] != total_data_frames - 1:
        frame_indices.append(total_data_frames - 1)

    frames: list[Image.Image] = []
    for frame_index in frame_indices:
        progress = frame_index / max(1, total_data_frames - 1)
        t = frame_index / data_fps
        image = Image.new("RGB", size, (6, 9, 14))
        draw = ImageDraw.Draw(image)

        draw_panel(draw, skeleton_rect, motion_title, title_font)
        draw_panel(draw, audio_rect, "Aligned sliced music", title_font)

        draw.text(
            (skeleton_rect[0] + 18, skeleton_rect[1] + 52),
            f"{key} | frame {frame_index + 1}/{total_data_frames} | {t:0.2f}s",
            fill=(169, 181, 198),
            font=tiny_font,
        )
        draw.text(
            (audio_rect[0] + 18, audio_rect[1] + 52),
            f"{audio_path.name} | {sample_rate} Hz | {duration:0.2f}s",
            fill=(169, 181, 198),
            font=tiny_font,
        )

        pose_2d = project_pose(
            positions,
            frame_index,
            (skeleton_rect[0] + 30, skeleton_rect[1] + 72, skeleton_rect[2] - 30, skeleton_rect[3] - 36),
            global_scale,
            position_source,
        )
        draw_skeleton(draw, pose_2d, skeleton_rect)
        draw_waveform(draw, waveform_rect, waveform_lo, waveform_hi, progress)
        draw.text((waveform_rect[0], waveform_rect[1] - 25), "waveform", fill=(203, 213, 228), font=small_font)
        draw_progress_on_spectrogram(image, spectro_rect, spectrogram, progress)
        draw = ImageDraw.Draw(image)
        draw.text((spectro_rect[0], spectro_rect[1] - 25), "spectrogram", fill=(203, 213, 228), font=small_font)

        frames.append(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = max(20, int(round(1000 / fps)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--encodings",
        type=Path,
        default=DEFAULT_DATA_ROOT / "data_and_model" / "conditions" / "normalized_all_test_data_01.pkl",
    )
    parser.add_argument("--key", default=None, help="AIST++ sliced key without .wav")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--data-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--condition-motion",
        action="store_true",
        help="Draw normalized DMD condition positions instead of raw AIST++ FK when raw motion is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encodings = load_encodings(args.encodings)
    audio_dirs = [args.data_root / "test" / "wavs_sliced", args.data_root / "train" / "wavs_sliced"]
    key = resolve_key(encodings, audio_dirs, args.key)
    audio_path = resolve_audio(audio_dirs, key)

    output_path = args.output_dir / f"{key}_motion_audio.gif"
    render_gif(
        key=key,
        encoding=np.asarray(encodings[key]),
        audio_path=audio_path,
        data_root=args.data_root,
        output_path=output_path,
        fps=args.fps,
        data_fps=args.data_fps,
        size=(args.width, args.height),
        prefer_raw=not args.condition_motion,
    )
    print(output_path)


if __name__ == "__main__":
    main()
