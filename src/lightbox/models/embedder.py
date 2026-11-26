"""SigLIP 2 Base ONNX lazy dual-tower embedder & zero-text-tower scene vocabulary (SPEC §6.2–6.3, PLAN T11)."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lightbox.paths import AppPaths, get_app_paths

EMBEDDING_DIM = 768
SIGLIP2_MODEL_ID = "siglip2-base-patch16-256"
SIGLIP2_DIRNAME = "siglip2-base"
SIGLIP2_DEFAULT_HF_REPO = "onnx-community/siglip2-base-patch16-256-ONNX"
SIGLIP2_VISION_FILENAME = "onnx/vision_model_int8.onnx"
SIGLIP2_TEXT_FILENAME = "onnx/text_model_int8.onnx"
SIGLIP2_TOKENIZER_FILENAME = "tokenizer.json"
SIGLIP2_IMAGE_SIZE = 256
SIGLIP2_MAX_TEXT_LENGTH = 64

SCENE_VOCAB_NPY_PATH = Path(__file__).resolve().parent / "data" / "scene_vocab.npy"


class EmbedderError(RuntimeError):
    """Raised when local SigLIP 2 ONNX embedding cannot be performed."""


@dataclass(frozen=True, slots=True)
class SceneVocabEntry:
    """Single precomputed photography scene or subject-motion vocabulary concept."""

    tag: str
    prompt: str
    label_zh: str
    label_en: str
    kind: Literal["scene", "motion"] = "scene"


# ~40 photography scene + subject motion concepts (SPEC §6.2–6.3, PLAN T11).
# Order MUST stay aligned with rows of `src/lightbox/models/data/scene_vocab.npy`.
SCENE_VOCAB: tuple[SceneVocabEntry, ...] = (
    # People & portraits
    SceneVocabEntry("portrait", "a photo of a portrait of a single person", "人像", "Portrait"),
    SceneVocabEntry("closeup_portrait", "a close-up headshot portrait of a face", "特写人像", "Close-up portrait"),
    SceneVocabEntry("group_portrait", "a photo of a group of people posing together", "多人合影", "Group portrait"),
    SceneVocabEntry("candid_people", "a candid street photo of people in daily life", "纪实人物", "Candid people"),
    SceneVocabEntry("wedding_event", "a photo of a wedding ceremony or celebration event", "婚礼活动", "Wedding / event"),
    SceneVocabEntry("child_family", "a photo of children or family moment", "亲子家庭", "Family / kids"),
    # Urban, street & architecture
    SceneVocabEntry("street", "a street photography photo of an urban road and pedestrians", "街拍", "Street"),
    SceneVocabEntry("cityscape", "a wide cityscape photo of buildings and skyline", "城市风光", "Cityscape"),
    SceneVocabEntry("architecture", "an architectural photo of a building facade or structure", "建筑", "Architecture"),
    SceneVocabEntry("night_city", "a night city photo with neon lights and illuminated streets", "城市夜景", "Night city"),
    SceneVocabEntry("travel_landmark", "a travel photo of a famous monument or historic landmark", "地标旅行", "Landmark"),
    SceneVocabEntry("vehicle_transport", "a photo of a car, train, motorcycle, or airplane", "交通工具", "Vehicle"),
    # Nature & landscape
    SceneVocabEntry("landscape_mountain", "a landscape photo of mountains and valleys", "山川风光", "Mountains"),
    SceneVocabEntry("beach_ocean", "a photo of a beach, ocean waves, and coastline", "海滨海岸", "Beach / ocean"),
    SceneVocabEntry("forest_nature", "a nature photo of a green forest and tall trees", "森林自然", "Forest"),
    SceneVocabEntry("lake_river", "a landscape photo of a calm lake, river, or waterfall", "湖泊河流", "Lake / river"),
    SceneVocabEntry("sunset_golden_hour", "a photo of a golden hour sunset or sunrise sky", "日出日落", "Sunset / sunrise"),
    SceneVocabEntry("blue_hour_twilight", "a twilight photo during blue hour after sunset", "蓝调时刻", "Blue hour"),
    SceneVocabEntry("night_sky_stars", "a night photography photo of stars and Milky Way sky", "星空银河", "Night sky"),
    SceneVocabEntry("snow_winter", "a winter landscape photo covered with snow and ice", "雪景冬日", "Winter snow"),
    SceneVocabEntry("rain_weather", "a moody photo taken in rain, fog, or mist", "雨雾天气", "Rain / fog"),
    SceneVocabEntry("reflection_water", "a photo featuring mirror reflections on water or glass", "倒影", "Reflection"),
    SceneVocabEntry("silhouette_backlit", "a backlit silhouette photo against bright light", "剪影逆光", "Silhouette"),
    # Flora, fauna & macro
    SceneVocabEntry("flower_botanical", "a close-up photo of flowers, blossoms, or leaves", "花卉植物", "Flowers"),
    SceneVocabEntry("macro_detail", "a macro close-up photo of tiny textures or insects", "微距细节", "Macro detail"),
    SceneVocabEntry("pet_animal", "a photo of a pet dog or cat", "宠物猫狗", "Pet"),
    SceneVocabEntry("wildlife_bird", "a wildlife photo of wild animals or birds in nature", "野生动物", "Wildlife"),
    # Indoor, food & culture
    SceneVocabEntry("food_drink", "a food photography photo of a meal, dessert, or coffee", "美食饮品", "Food / drink"),
    SceneVocabEntry("cafe_restaurant", "an indoor photo inside a cozy cafe or restaurant", "咖啡馆餐厅", "Cafe / dining"),
    SceneVocabEntry("indoor_home", "an indoor photo of a living room or home interior", "室内居家", "Indoor / home"),
    SceneVocabEntry("museum_art", "a photo inside a museum, gallery, or exhibition", "展览艺术", "Museum / art"),
    SceneVocabEntry("concert_stage", "a live concert, theater, or stage performance with spotlights", "演出舞台", "Stage / concert"),
    SceneVocabEntry("studio_product", "a clean studio still-life or product photo", "静物产品", "Still life"),
    SceneVocabEntry("document_text", "a photo of a document, book page, sign, or screen", "文字文档", "Document / sign"),
    # Subject motion & dynamic cues (SPEC §6.2)
    SceneVocabEntry("action_running", "a dynamic action photo of a person running or jumping", "奔跑跳跃", "Running / jumping", kind="motion"),
    SceneVocabEntry("sports_motion", "a fast-paced sports action photo of athletes in motion", "运动瞬间", "Sports action", kind="motion"),
    SceneVocabEntry("dancing_motion", "a photo of a person dancing or performing expressive movement", "舞蹈律动", "Dancing", kind="motion"),
    SceneVocabEntry("panning_speed", "a motion panning photo of a fast moving vehicle or cyclist", "追焦动感", "Panning motion", kind="motion"),
    SceneVocabEntry("flying_motion", "an action photo of a bird or object flying mid-air", "飞翔瞬间", "Mid-air motion", kind="motion"),
    SceneVocabEntry("long_exposure_trails", "a long exposure photo with light trails or silky flowing water", "长曝流光", "Long exposure", kind="motion"),
    SceneVocabEntry("static_still_scene", "a calm, completely still and static scene with no moving subjects", "静止画面", "Static scene", kind="motion"),
    SceneVocabEntry("monochrome_bw", "a black and white monochrome artistic photograph", "黑白单色", "Monochrome B&W"),
)

_SCENE_ENTRY_BY_TAG: dict[str, SceneVocabEntry] = {entry.tag: entry for entry in SCENE_VOCAB}


def get_scene_label(tag: str | None, *, lang: str = "zh-CN") -> str | None:
    """Return localized human-readable label for a `scene_tag` slug."""
    if not tag:
        return None
    entry = _SCENE_ENTRY_BY_TAG.get(tag)
    if entry is None:
        return tag.replace("_", " ")
    return entry.label_zh if lang.startswith("zh") else entry.label_en


@runtime_checkable
class Embedder(Protocol):
    """Protocol for image/text semantic embedders (real SigLIP 2 ONNX or test fakes)."""

    model_id: str
    dim: int

    def embed_images(self, images: Sequence[Image.Image | Path]) -> NDArray[np.float32]: ...

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


def _l2_normalize(mat: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return (arr / norm).astype(np.float32, copy=False) if norm > 0.0 else arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return (arr / np.maximum(norms, np.float32(1e-12))).astype(np.float32, copy=False)


def preprocess_siglip2_image(image: Image.Image | Path) -> NDArray[np.float32]:
    """Resize to 256x256 RGB and normalize to [-1, 1] CHW float32 per SigLIP 2 preprocessor_config."""
    if isinstance(image, Path):
        with Image.open(image) as opened:
            rgb = opened.convert("RGB")
            resized = rgb.resize((SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE), Image.Resampling.BILINEAR)
    else:
        rgb = image.convert("RGB")
        resized = rgb.resize((SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    arr = (arr - np.float32(0.5)) / np.float32(0.5)
    return np.transpose(arr, (2, 0, 1))


def find_siglip2_vision_path(
    app_paths: AppPaths | None = None,
    *,
    vision_filename: str = SIGLIP2_VISION_FILENAME,
) -> Path | None:
    """Return local cached path to SigLIP 2 vision ONNX file if already downloaded."""
    paths = app_paths or get_app_paths()
    candidate = paths.models_dir / SIGLIP2_DIRNAME / vision_filename
    return candidate if candidate.is_file() else None


def find_siglip2_text_path(
    app_paths: AppPaths | None = None,
    *,
    text_filename: str = SIGLIP2_TEXT_FILENAME,
    tokenizer_filename: str = SIGLIP2_TOKENIZER_FILENAME,
) -> tuple[Path | None, Path | None]:
    """Return local cached paths to `(text_model.onnx, tokenizer.json)` if already downloaded."""
    paths = app_paths or get_app_paths()
    base = paths.models_dir / SIGLIP2_DIRNAME
    text_p = base / text_filename
    tok_p = base / tokenizer_filename
    return (text_p if text_p.is_file() else None, tok_p if tok_p.is_file() else None)


def _download_hf_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a single model file via `huggingface_hub` (automatically honors `HF_ENDPOINT`)."""
    try:
        from huggingface_hub import hf_hub_download

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
        return Path(downloaded)
    except EmbedderError:
        raise
    except Exception as exc:
        raise EmbedderError(
            f"Failed to download SigLIP 2 model file {filename!r} from {repo_id!r} "
            f"(check network connection or HF_ENDPOINT): {exc}"
        ) from exc


def _configure_siglip2_tokenizer(tokenizer: Any) -> tuple[int, int]:
    """Configure fixed-length truncation/padding once on tokenizer load for thread-safe encoding."""
    pad_id = tokenizer.token_to_id("<pad>")
    pad_val = int(pad_id) if pad_id is not None else 0
    eos_id = tokenizer.token_to_id("<eos>")
    eos_val = int(eos_id) if eos_id is not None else 1
    tokenizer.enable_truncation(max_length=SIGLIP2_MAX_TEXT_LENGTH)
    tokenizer.enable_padding(
        pad_id=pad_val,
        pad_token="<pad>",
        length=SIGLIP2_MAX_TEXT_LENGTH,
    )
    return pad_val, eos_val


def _encode_siglip2_tokens(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    pad_val: int = 0,
    eos_val: int = 1,
) -> NDArray[np.int64]:
    """Tokenize texts for SigLIP 2 (lowercased, EOS-terminated, padded/truncated to 64)."""
    rows: list[list[int]] = []
    for raw in texts:
        text = raw.strip().lower()
        enc = tokenizer.encode(text)
        ids = [int(x) for x in enc.ids[:SIGLIP2_MAX_TEXT_LENGTH]]
        if eos_val not in ids:
            # Ensure <eos> terminates non-pad tokens if custom tokenizer did not append it
            non_pad_len = len(ids)
            while non_pad_len > 0 and ids[non_pad_len - 1] == pad_val:
                non_pad_len -= 1
            insert_pos = min(non_pad_len, SIGLIP2_MAX_TEXT_LENGTH - 1)
            ids[insert_pos] = eos_val
        if len(ids) < SIGLIP2_MAX_TEXT_LENGTH:
            ids.extend([pad_val] * (SIGLIP2_MAX_TEXT_LENGTH - len(ids)))
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


class Siglip2Embedder:
    """Lazy dual-tower SigLIP 2 Base ONNX embedder (`vision` and `text` towers strictly split).

    - Indexing / scene-tagging only ever downloads & loads the vision tower (`vision_model_int8.onnx`).
    - Text query embedding lazily downloads & loads the text tower (`text_model_int8.onnx` + `tokenizer.json`).
    - Downloads use `huggingface_hub.hf_hub_download` into `<APP_DATA_DIR>/models/siglip2-base/`
      and automatically respect `HF_ENDPOINT`.
    """

    model_id: str = SIGLIP2_MODEL_ID
    dim: int = EMBEDDING_DIM

    def __init__(
        self,
        app_paths: AppPaths | None = None,
        *,
        repo_id: str | None = None,
        vision_filename: str = SIGLIP2_VISION_FILENAME,
        text_filename: str = SIGLIP2_TEXT_FILENAME,
        tokenizer_filename: str = SIGLIP2_TOKENIZER_FILENAME,
        allow_download: bool = True,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.app_paths = app_paths or get_app_paths()
        self.repo_id = (
            repo_id
            or os.environ.get("LIGHTBOX_SIGLIP2_REPO")
            or SIGLIP2_DEFAULT_HF_REPO
        )
        self.vision_filename = vision_filename
        self.text_filename = text_filename
        self.tokenizer_filename = tokenizer_filename
        self.allow_download = allow_download
        self.providers = list(providers)

        self._vision_session: Any = None
        self._text_session: Any = None
        self._tokenizer: Any = None
        self._pad_val: int = 0
        self._eos_val: int = 1
        self._vision_lock = threading.Lock()
        self._text_lock = threading.Lock()

    @property
    def model_dir(self) -> Path:
        return self.app_paths.models_dir / SIGLIP2_DIRNAME

    @property
    def vision_loaded(self) -> bool:
        return self._vision_session is not None

    @property
    def text_loaded(self) -> bool:
        return self._text_session is not None

    def _resolve_vision_path(self) -> Path:
        existing = find_siglip2_vision_path(
            self.app_paths,
            vision_filename=self.vision_filename,
        )
        if existing is not None:
            return existing
        if not self.allow_download:
            raise EmbedderError(
                f"SigLIP 2 vision model not found under {self.model_dir} and downloads are disabled."
            )
        return _download_hf_file(self.repo_id, self.vision_filename, self.model_dir)

    def _resolve_text_paths(self) -> tuple[Path, Path]:
        text_p, tok_p = find_siglip2_text_path(
            self.app_paths,
            text_filename=self.text_filename,
            tokenizer_filename=self.tokenizer_filename,
        )
        if text_p is None:
            if not self.allow_download:
                raise EmbedderError(
                    f"SigLIP 2 text model not found under {self.model_dir} and downloads are disabled."
                )
            text_p = _download_hf_file(self.repo_id, self.text_filename, self.model_dir)
        if tok_p is None:
            if not self.allow_download:
                raise EmbedderError(
                    f"SigLIP 2 tokenizer not found under {self.model_dir} and downloads are disabled."
                )
            tok_p = _download_hf_file(self.repo_id, self.tokenizer_filename, self.model_dir)
        return text_p, tok_p

    def _ensure_vision_session(self) -> Any:
        if self._vision_session is not None:
            return self._vision_session
        with self._vision_lock:
            if self._vision_session is not None:
                return self._vision_session
            import onnxruntime as ort

            model_path = self._resolve_vision_path()
            try:
                self._vision_session = ort.InferenceSession(
                    str(model_path), providers=self.providers
                )
            except Exception as exc:
                raise EmbedderError(
                    f"Failed to load SigLIP 2 vision ONNX model at {model_path}: {exc}"
                ) from exc
            return self._vision_session

    def _ensure_text_session(self) -> tuple[Any, Any]:
        if self._text_session is not None and self._tokenizer is not None:
            return self._text_session, self._tokenizer
        with self._text_lock:
            if self._text_session is not None and self._tokenizer is not None:
                return self._text_session, self._tokenizer
            import onnxruntime as ort
            from tokenizers import Tokenizer

            text_path, tok_path = self._resolve_text_paths()
            try:
                tokenizer = Tokenizer.from_file(str(tok_path))
                pad_val, eos_val = _configure_siglip2_tokenizer(tokenizer)
                session = ort.InferenceSession(str(text_path), providers=self.providers)
            except Exception as exc:
                raise EmbedderError(
                    f"Failed to load SigLIP 2 text ONNX model/tokenizer at {text_path}: {exc}"
                ) from exc
            self._pad_val = pad_val
            self._eos_val = eos_val
            self._tokenizer = tokenizer
            self._text_session = session
            return self._text_session, self._tokenizer

    def embed_images(
        self,
        images: Sequence[Image.Image | Path],
        *,
        batch_size: int = 8,
    ) -> NDArray[np.float32]:
        """Encode one or more images into L2-normalized `(N, 768)` float32 vectors (vision tower only)."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        session = self._ensure_vision_session()
        input_name = str(session.get_inputs()[0].name)
        output_names = [str(o.name) for o in session.get_outputs()]
        pool_name = "pooler_output" if "pooler_output" in output_names else output_names[-1]

        chunks: list[NDArray[np.float32]] = []
        step = max(1, batch_size)
        for start in range(0, len(images), step):
            batch = np.stack(
                [preprocess_siglip2_image(im) for im in images[start : start + step]],
                axis=0,
            ).astype(np.float32, copy=False)
            raw = session.run([pool_name], {input_name: batch})[0]
            chunks.append(_l2_normalize(np.asarray(raw, dtype=np.float32)))
        return np.concatenate(chunks, axis=0)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """Encode one or more Chinese/English texts into L2-normalized `(N, 768)` float32 vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        session, tokenizer = self._ensure_text_session()
        input_name = str(session.get_inputs()[0].name)
        output_names = [str(o.name) for o in session.get_outputs()]
        pool_name = "pooler_output" if "pooler_output" in output_names else output_names[-1]

        chunks: list[NDArray[np.float32]] = []
        step = max(1, batch_size)
        for start in range(0, len(texts), step):
            ids = _encode_siglip2_tokens(
                tokenizer,
                texts[start : start + step],
                pad_val=self._pad_val,
                eos_val=self._eos_val,
            )
            raw = session.run([pool_name], {input_name: ids})[0]
            chunks.append(_l2_normalize(np.asarray(raw, dtype=np.float32)))
        return np.concatenate(chunks, axis=0)


_EMBEDDER_CACHE_LOCK = threading.Lock()
_EMBEDDER_CACHE: dict[tuple[str, str, str, str, str, bool, tuple[str, ...]], Siglip2Embedder] = {}


def get_shared_siglip2_embedder(
    app_paths: AppPaths | None = None,
    *,
    repo_id: str | None = None,
    vision_filename: str = SIGLIP2_VISION_FILENAME,
    text_filename: str = SIGLIP2_TEXT_FILENAME,
    tokenizer_filename: str = SIGLIP2_TOKENIZER_FILENAME,
    allow_download: bool = True,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> Siglip2Embedder:
    """Return a process-wide shared `Siglip2Embedder` so vision/text ONNX sessions load at most once."""
    paths = app_paths or get_app_paths()
    resolved_repo = (
        repo_id or os.environ.get("LIGHTBOX_SIGLIP2_REPO") or SIGLIP2_DEFAULT_HF_REPO
    )
    key = (
        str(paths.models_dir.resolve()),
        resolved_repo,
        vision_filename,
        text_filename,
        tokenizer_filename,
        allow_download,
        tuple(providers),
    )
    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is None:
            cached = Siglip2Embedder(
                paths,
                repo_id=resolved_repo,
                vision_filename=vision_filename,
                text_filename=text_filename,
                tokenizer_filename=tokenizer_filename,
                allow_download=allow_download,
                providers=providers,
            )
            _EMBEDDER_CACHE[key] = cached
        return cached


def clear_shared_siglip2_embedder_cache() -> None:
    """Clear the process-level shared `Siglip2Embedder` cache (primarily for tests)."""
    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


_CACHED_SCENE_VOCAB_MATRIX: NDArray[np.float32] | None = None


def load_scene_vocab(
    npy_path: Path | None = None,
) -> tuple[tuple[SceneVocabEntry, ...], NDArray[np.float32]]:
    """Load bundled precomputed scene/motion prompt vectors `(K, 768)` without touching the text tower."""
    global _CACHED_SCENE_VOCAB_MATRIX
    if npy_path is None and _CACHED_SCENE_VOCAB_MATRIX is not None:
        return SCENE_VOCAB, _CACHED_SCENE_VOCAB_MATRIX

    path = npy_path or SCENE_VOCAB_NPY_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Precomputed scene vocabulary matrix not found at {path}. "
            "Run `uv run python scripts/build_scene_vocab.py` to generate it."
        )
    matrix = np.asarray(np.load(path), dtype=np.float32)
    expected_shape = (len(SCENE_VOCAB), EMBEDDING_DIM)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Scene vocabulary matrix shape mismatch: expected {expected_shape}, got {matrix.shape}."
        )
    normalized = _l2_normalize(matrix)
    if npy_path is None:
        _CACHED_SCENE_VOCAB_MATRIX = normalized
    return SCENE_VOCAB, normalized


def classify_scene(
    image_embedding: NDArray[np.float32] | Sequence[float],
    *,
    vocab_matrix: NDArray[np.float32] | None = None,
    entries: Sequence[SceneVocabEntry] = SCENE_VOCAB,
    kind: Literal["scene", "motion"] | None = "scene",
) -> tuple[str, float]:
    """Assign top `scene_tag` by dot product against precomputed `.npy` vectors (zero text tower)."""
    vec = _l2_normalize(np.asarray(image_embedding, dtype=np.float32).reshape(-1))
    if vec.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim embedding, got {vec.shape[0]}.")
    mat = vocab_matrix if vocab_matrix is not None else load_scene_vocab()[1]
    scores = mat @ vec
    if kind is not None:
        candidate_indices = [i for i, e in enumerate(entries) if e.kind == kind]
        if candidate_indices:
            best_idx = max(candidate_indices, key=lambda idx: float(scores[idx]))
            return entries[best_idx].tag, float(scores[best_idx])
    best_idx = int(np.argmax(scores))
    return entries[best_idx].tag, float(scores[best_idx])


__all__ = [
    "EMBEDDING_DIM",
    "SCENE_VOCAB",
    "SCENE_VOCAB_NPY_PATH",
    "SIGLIP2_DEFAULT_HF_REPO",
    "SIGLIP2_DIRNAME",
    "SIGLIP2_IMAGE_SIZE",
    "SIGLIP2_MAX_TEXT_LENGTH",
    "SIGLIP2_MODEL_ID",
    "SIGLIP2_TEXT_FILENAME",
    "SIGLIP2_TOKENIZER_FILENAME",
    "SIGLIP2_VISION_FILENAME",
    "Embedder",
    "EmbedderError",
    "SceneVocabEntry",
    "Siglip2Embedder",
    "classify_scene",
    "clear_shared_siglip2_embedder_cache",
    "find_siglip2_text_path",
    "find_siglip2_vision_path",
    "get_scene_label",
    "get_shared_siglip2_embedder",
    "load_scene_vocab",
    "preprocess_siglip2_image",
]
