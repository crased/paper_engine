"""
Scene Context — structured visual descriptions for YOLO annotation grounding.

Pairs human-written natural language with LLM-extracted structured descriptors
so the batch annotator knows exactly what tangible objects look like per scene.

Flow:
    1. User writes a free-text description while viewing a sample frame
    2. LLM (Gemini) + the sample image extracts structured ObjectDescriptors
    3. User reviews / edits the structured fields
    4. Saved to JSON per game (conf/game_configs/<game>_scenes.json)
    5. batch_annotator loads scene context and injects it into annotation prompts

Storage format (JSON):
    {
        "scene_level_frogs": {
            "display_name": "Ribby & Croaks",
            "scene_type": "boss",
            "raw_description": "Big green frog on the right...",
            "player": {"name": "Cuphead", "appearance": "...", ...},
            "enemies": [...],
            "projectiles": [...],
            "hazards": [...],
            "platforms": [...],
            "terrain": "flat arena, no platforms",
            "sample_image": "recordings/sessions/.../frame_000042.png"
        },
        ...
    }

Usage:
    from pipeline.scene_context import SceneContextStore, SceneContext

    store = SceneContextStore("Cuphead")
    ctx = store.get("scene_level_frogs")
    prompt_block = ctx.to_prompt_block()   # ready for injection into annotator
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ObjectDescriptor:
    """A tangible visual object the annotator should look for."""

    name: str = ""
    appearance: str = ""  # color, shape, distinguishing visual features
    size: str = ""  # relative to player or approx pixel dimensions
    position: str = ""  # typical screen location (left/right/ground/air)
    notes: str = ""  # anything else (movement, phases, what it blends into)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectDescriptor":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_prompt_line(self) -> str:
        """Single-line description for injection into annotation prompt."""
        parts = []
        if self.name:
            parts.append(self.name)
        if self.appearance:
            parts.append(self.appearance)
        if self.size:
            parts.append(f"size: {self.size}")
        if self.position:
            parts.append(f"position: {self.position}")
        if self.notes:
            parts.append(self.notes)
        return ", ".join(parts)


@dataclass
class SceneContext:
    """Structured visual context for one game scene."""

    scene_name: str  # from memory reading, e.g. "scene_level_frogs"
    display_name: str = ""  # human-friendly, e.g. "Ribby & Croaks"
    scene_type: str = ""  # boss / run_and_gun / platforming / menu / map
    raw_description: str = ""  # original free-text the user wrote

    player: Optional[ObjectDescriptor] = None
    enemies: List[ObjectDescriptor] = field(default_factory=list)
    projectiles: List[ObjectDescriptor] = field(default_factory=list)
    hazards: List[ObjectDescriptor] = field(default_factory=list)
    platforms: List[ObjectDescriptor] = field(default_factory=list)
    terrain: str = ""

    sample_image: str = ""  # relative path to a representative frame

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "display_name": self.display_name,
            "scene_type": self.scene_type,
            "raw_description": self.raw_description,
            "player": self.player.to_dict() if self.player else None,
            "enemies": [e.to_dict() for e in self.enemies],
            "projectiles": [p.to_dict() for p in self.projectiles],
            "hazards": [h.to_dict() for h in self.hazards],
            "platforms": [p.to_dict() for p in self.platforms],
            "terrain": self.terrain,
            "sample_image": self.sample_image,
        }
        return d

    @classmethod
    def from_dict(cls, scene_name: str, d: dict) -> "SceneContext":
        return cls(
            scene_name=scene_name,
            display_name=d.get("display_name", ""),
            scene_type=d.get("scene_type", ""),
            raw_description=d.get("raw_description", ""),
            player=ObjectDescriptor.from_dict(d["player"]) if d.get("player") else None,
            enemies=[ObjectDescriptor.from_dict(e) for e in d.get("enemies", [])],
            projectiles=[ObjectDescriptor.from_dict(p) for p in d.get("projectiles", [])],
            hazards=[ObjectDescriptor.from_dict(h) for h in d.get("hazards", [])],
            platforms=[ObjectDescriptor.from_dict(p) for p in d.get("platforms", [])],
            terrain=d.get("terrain", ""),
            sample_image=d.get("sample_image", ""),
        )

    def to_prompt_block(self) -> str:
        """Format as a text block for injection into the annotation prompt.

        Example output:
            Scene: scene_level_frogs ("Ribby & Croaks", boss)
              Player: Cuphead, small cup character with blue shorts, ~80x100px, left side ground level
              Enemy: Ribby, large green frog, 2x player size, right side
              Projectile: pink fireflies, small ~30px, arc from right to left
              Terrain: flat arena, no platforms
        """
        lines = []
        header = f"Scene: {self.scene_name}"
        if self.display_name:
            header += f' ("{self.display_name}"'
            if self.scene_type:
                header += f", {self.scene_type}"
            header += ")"
        lines.append(header)

        if self.player:
            lines.append(f"  Player: {self.player.to_prompt_line()}")
        for e in self.enemies:
            lines.append(f"  Enemy: {e.to_prompt_line()}")
        for p in self.projectiles:
            lines.append(f"  Projectile: {p.to_prompt_line()}")
        for h in self.hazards:
            lines.append(f"  Hazard: {h.to_prompt_line()}")
        for p in self.platforms:
            lines.append(f"  Platform: {p.to_prompt_line()}")
        if self.terrain:
            lines.append(f"  Terrain: {self.terrain}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistent store (one JSON file per game)
# ---------------------------------------------------------------------------


class SceneContextStore:
    """Load / save scene context entries for a game.

    Storage path: conf/game_configs/<game_lower>_scenes.json
    """

    def __init__(self, game_name: str):
        self.game_name = game_name
        safe = game_name.lower().replace(" ", "_")
        self._path = _PROJECT_ROOT / "conf" / "game_configs" / f"{safe}_scenes.json"
        self._contexts: Dict[str, SceneContext] = {}
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Dict[str, SceneContext]:
        """Load all scene contexts from JSON file."""
        self._contexts = {}
        if not self._path.exists():
            return self._contexts
        try:
            raw = json.loads(self._path.read_text())
            for scene_name, data in raw.items():
                self._contexts[scene_name] = SceneContext.from_dict(scene_name, data)
        except Exception as e:
            logger.error("Failed to load scene contexts from %s: %s", self._path, e)
        return self._contexts

    def save(self):
        """Write all scene contexts to JSON file."""
        data = {name: ctx.to_dict() for name, ctx in self._contexts.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved %d scene contexts to %s", len(data), self._path)

    def get(self, scene_name: str) -> Optional[SceneContext]:
        return self._contexts.get(scene_name)

    def set(self, ctx: SceneContext):
        self._contexts[ctx.scene_name] = ctx

    def delete(self, scene_name: str):
        self._contexts.pop(scene_name, None)

    def all(self) -> Dict[str, SceneContext]:
        return dict(self._contexts)

    def scene_names(self) -> List[str]:
        return sorted(self._contexts.keys())


# ---------------------------------------------------------------------------
# Scene discovery — scan recorded sessions for unique scene names + sample frames
# ---------------------------------------------------------------------------


def discover_scenes(sessions_root: Optional[Path] = None) -> Dict[str, str]:
    """Scan recorded session frames for unique scene_name values.

    Returns:
        dict mapping scene_name -> path to first frame PNG where that scene appeared
    """
    if sessions_root is None:
        sessions_root = _PROJECT_ROOT / "recordings" / "sessions"

    scenes: Dict[str, str] = {}  # scene_name -> sample image path

    if not sessions_root.exists():
        return scenes

    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue
        for frame_json in sorted(session_dir.glob("frame_*.json")):
            try:
                data = json.loads(frame_json.read_text())
                scene = data.get("state", {}).get("scene_name", "")
                if scene and scene not in scenes:
                    # Use corresponding PNG as sample image
                    frame_png = frame_json.with_suffix(".png")
                    if frame_png.exists():
                        scenes[scene] = str(frame_png)
            except Exception:
                continue

    return scenes


# ---------------------------------------------------------------------------
# LLM extraction — natural language + image → structured SceneContext
# ---------------------------------------------------------------------------


_EXTRACTION_PROMPT = """\
You are analyzing a game screenshot along with a human-written description of what's visible.
Extract structured visual information about the tangible objects in this scene.

HUMAN DESCRIPTION:
{raw_description}

Extract the following as JSON. Only include objects the human mentioned or that you can
clearly see in the image. Do NOT invent objects that aren't described or visible.

Return ONLY valid JSON (no markdown fences):
{{
    "display_name": "human-friendly scene name if mentioned, else empty string",
    "scene_type": "boss / run_and_gun / platforming / menu / map / other",
    "player": {{
        "name": "character name",
        "appearance": "visual description — color, shape, distinguishing features",
        "size": "approximate size relative to screen or in pixels",
        "position": "typical screen position",
        "notes": ""
    }},
    "enemies": [
        {{
            "name": "enemy name or type",
            "appearance": "visual description",
            "size": "relative or absolute size",
            "position": "where on screen",
            "notes": "movement patterns, phase changes, what it blends into"
        }}
    ],
    "projectiles": [
        {{
            "name": "projectile type",
            "appearance": "color, shape",
            "size": "approximate size",
            "position": "trajectory or spawn location",
            "notes": ""
        }}
    ],
    "hazards": [
        {{
            "name": "hazard type",
            "appearance": "visual description",
            "size": "",
            "position": "where on screen",
            "notes": ""
        }}
    ],
    "platforms": [
        {{
            "name": "platform type",
            "appearance": "visual description",
            "size": "",
            "position": "where on screen",
            "notes": ""
        }}
    ],
    "terrain": "general terrain / ground description"
}}

Rules:
- If the human didn't mention a category and you can't see it, use null for player or empty arrays.
- Be precise about visual features: colors, shapes, relative sizes.
- For position, use screen-relative terms: left/right/center, top/bottom/ground/air.
- For size, relate to player size or give approximate pixel dimensions.
- Combine the human's description with what you observe in the image.
"""


def _parse_json_response(text: str) -> dict:
    """Extract JSON object from LLM response, stripping markdown fences if present."""
    import re

    cleaned = text.strip()
    # Strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    return json.loads(cleaned)


def _build_scene_context_from_parsed(
    data: dict, scene_name: str, raw_description: str
) -> SceneContext:
    """Convert parsed LLM JSON into a SceneContext dataclass."""
    player = None
    if data.get("player"):
        player = ObjectDescriptor.from_dict(data["player"])

    return SceneContext(
        scene_name=scene_name,
        display_name=data.get("display_name", ""),
        scene_type=data.get("scene_type", ""),
        raw_description=raw_description,
        player=player,
        enemies=[ObjectDescriptor.from_dict(e) for e in data.get("enemies") or []],
        projectiles=[ObjectDescriptor.from_dict(p) for p in data.get("projectiles") or []],
        hazards=[ObjectDescriptor.from_dict(h) for h in data.get("hazards") or []],
        platforms=[ObjectDescriptor.from_dict(p) for p in data.get("platforms") or []],
        terrain=data.get("terrain", ""),
    )


def extract_scene_context(
    raw_description: str,
    sample_image_path: Optional[str] = None,
    scene_name: str = "",
) -> SceneContext:
    """Send raw description + sample image to LLM, return structured SceneContext.

    Falls back to an empty SceneContext with just raw_description if LLM unavailable.
    """
    from pipeline.auto_review import _get_llm_client, _call_llm_vision

    fallback = SceneContext(scene_name=scene_name, raw_description=raw_description)

    try:
        client, provider, model = _get_llm_client()
    except Exception as exc:
        logger.error("LLM client unavailable: %s", exc)
        return fallback

    prompt = _EXTRACTION_PROMPT.format(raw_description=raw_description)

    # Load sample image if provided
    images = []
    if sample_image_path:
        try:
            img = Image.open(sample_image_path)
            img.load()
            images.append(img)
        except Exception as exc:
            logger.warning("Could not load sample image %s: %s", sample_image_path, exc)

    try:
        if images:
            response = _call_llm_vision(client, provider, model, prompt, images, max_tokens=4096)
        else:
            # Text-only fallback (no image)
            from pipeline.generate_bot_script import call_llm
            response = call_llm(client, provider, model, prompt, max_tokens=4096)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return fallback

    try:
        data = _parse_json_response(response)
        return _build_scene_context_from_parsed(data, scene_name, raw_description)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Failed to parse LLM response: %s\nRaw: %s", exc, response[:500])
        return fallback


# ---------------------------------------------------------------------------
# Prompt integration helper (used by batch_annotator)
# ---------------------------------------------------------------------------

# Module-level store cache (lazy-loaded)
_store_cache: Dict[str, SceneContextStore] = {}


def get_scene_prompt_block(game_name: str, scene_name: str) -> Optional[str]:
    """Look up scene context and return a formatted prompt block, or None.

    Called by batch_annotator._build_annotation_prompt() to enrich the
    annotation prompt with per-scene visual descriptions.
    """
    if game_name not in _store_cache:
        _store_cache[game_name] = SceneContextStore(game_name)
    ctx = _store_cache[game_name].get(scene_name)
    if ctx is None:
        return None
    return ctx.to_prompt_block()
