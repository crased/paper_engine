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
