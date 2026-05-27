"""
Cuphead game state configuration for the generic GameStateReader.

Defines all pointer chains needed to read Cuphead's game state via
external Mono memory reading. This replaces the hardcoded pointer chains
that were previously in cuphead_memory.py.

Cuphead specifics:
  - Unity 2017.4, Mono runtime with Boehm GC (objects don't move)
  - Assembly: "Assembly-CSharp" (5643 classes)
  - Key classes: PlayerManager, PlayerStatsManager, Level, SceneLoader,
    PlayerData, AbstractPlayerController

Pointer chains (discovered via reverse engineering):
  - HP chain: PlayerManager.players (static Dict<PlayerId,APC>)
              -> valueSlots[first non-null] (AbstractPlayerController)
              -> _stats (PlayerStatsManager instance)
              -> <Health>k__BackingField (int)
  - Level info: Level.<Current>k__BackingField (static ptr to instance)
              -> instance fields (LevelTime, Ending)
  - Level mode: Level.<CurrentMode>k__BackingField (static int/enum)
  - Scene: SceneLoader.<SceneName>k__BackingField (static string)
"""

from tools.game_state import StateFieldDef, ChainStep


# Game metadata
GAME_NAME = "Cuphead"
ASSEMBLY = "Assembly-CSharp"
GAME_EXE = "Cuphead.exe"
MONO_DLL = "mono.dll"


# ---------------------------------------------------------------------------
# State field definitions — each one is a named value with a pointer chain
# ---------------------------------------------------------------------------

FIELDS: list[StateFieldDef] = [
    # -- Player stats (via PlayerManager.players dict -> APC -> _stats -> PSM) --
    StateFieldDef(
        name="hp",
        read_type="int",
        chain=[
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(index=0, max_scan=4),
            ChainStep.field("_stats", owner_class="AbstractPlayerController"),
            ChainStep.field(
                "<Health>k__BackingField", owner_class="PlayerStatsManager"
            ),
        ],
        description="Current player health points",
    ),
    StateFieldDef(
        name="hp_max",
        read_type="int",
        chain=[
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(index=0, max_scan=4),
            ChainStep.field("_stats", owner_class="AbstractPlayerController"),
            ChainStep.field(
                "<HealthMax>k__BackingField", owner_class="PlayerStatsManager"
            ),
        ],
        description="Maximum player health points",
    ),
    StateFieldDef(
        name="super_meter",
        read_type="float",
        chain=[
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(index=0, max_scan=4),
            ChainStep.field("_stats", owner_class="AbstractPlayerController"),
            ChainStep.field(
                "<SuperMeter>k__BackingField", owner_class="PlayerStatsManager"
            ),
        ],
        description="Current super meter value",
    ),
    StateFieldDef(
        name="super_meter_max",
        read_type="float",
        chain=[
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(index=0, max_scan=4),
            ChainStep.field("_stats", owner_class="AbstractPlayerController"),
            ChainStep.field(
                "<SuperMeterMax>k__BackingField", owner_class="PlayerStatsManager"
            ),
        ],
        description="Maximum super meter value",
    ),
    StateFieldDef(
        name="deaths",
        read_type="int",
        chain=[
            ChainStep.static("PlayerManager", "players"),
            ChainStep.dict_value(index=0, max_scan=4),
            ChainStep.field("_stats", owner_class="AbstractPlayerController"),
            ChainStep.field(
                "<Deaths>k__BackingField", owner_class="PlayerStatsManager"
            ),
        ],
        description="Player death count for current session",
    ),
    # -- Level info (instance fields via Level.Current) --
    StateFieldDef(
        name="level_time",
        read_type="float",
        chain=[
            ChainStep.static("Level", "<Current>k__BackingField"),
            ChainStep.field("<LevelTime>k__BackingField"),
        ],
        description="Level timer in seconds",
    ),
    StateFieldDef(
        name="level_ending",
        read_type="bool",
        chain=[
            ChainStep.static("Level", "<Current>k__BackingField"),
            ChainStep.field("<Ending>k__BackingField"),
        ],
        description="Whether the level is ending (death or win animation)",
    ),
    # -- Level info (static fields — read directly from Level's static data) --
    StateFieldDef(
        name="level_won",
        read_type="bool",
        chain=[
            ChainStep.static("Level", "<Won>k__BackingField"),
        ],
        description="Whether the level was won",
    ),
    StateFieldDef(
        name="level_mode",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<CurrentMode>k__BackingField"),
        ],
        default=-1,
        description="Level mode: 0=Easy, 1=Normal, 2=Hard",
    ),
    # -- Game state --
    StateFieldDef(
        name="in_game",
        read_type="bool",
        chain=[
            ChainStep.static("PlayerData", "inGame"),
        ],
        description="Whether a save file is loaded",
    ),
    StateFieldDef(
        name="scene_name",
        read_type="string",
        chain=[
            ChainStep.static("SceneLoader", "<SceneName>k__BackingField"),
        ],
        description="Current scene name",
    ),
    StateFieldDef(
        name="is_loading",
        read_type="bool",
        chain=[
            ChainStep.static("SceneLoader", "currentlyLoading"),
        ],
        description="Whether a scene is being loaded",
    ),
    # -- Scoring data (static on Level, accessed by autosplitter) --
    StateFieldDef(
        name="num_times_hit",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<ScoringData>k__BackingField"),
            ChainStep.field("numTimesHit", owner_class="ScoringData"),
        ],
        description="Number of times player was hit this level attempt",
    ),
    StateFieldDef(
        name="num_parries",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<ScoringData>k__BackingField"),
            ChainStep.field("numParries", owner_class="ScoringData"),
        ],
        description="Number of parries performed this level attempt",
    ),
    StateFieldDef(
        name="super_meter_used",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<ScoringData>k__BackingField"),
            ChainStep.field("superMeterUsed", owner_class="ScoringData"),
        ],
        description="Super meter cards used this level attempt",
    ),
    StateFieldDef(
        name="coins_collected",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<ScoringData>k__BackingField"),
            ChainStep.field("coinsCollected", owner_class="ScoringData"),
        ],
        description="Coins collected this level attempt",
    ),
    StateFieldDef(
        name="level_grade",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<Grade>k__BackingField"),
        ],
        description="Level grade enum value",
    ),
    StateFieldDef(
        name="previous_level",
        read_type="int",
        chain=[
            ChainStep.static("Level", "<PreviousLevel>k__BackingField"),
        ],
        description="Previous level enum value (used by autosplitter for splits)",
    ),
]


# ---------------------------------------------------------------------------
# Summary / display helpers (optional, game-specific formatting)
# ---------------------------------------------------------------------------

MODE_NAMES = {0: "Easy", 1: "Normal", 2: "Hard"}


# ---------------------------------------------------------------------------
# Detection class ontology — for YOLO auto-labeling (game-specific)
# ---------------------------------------------------------------------------

# Class names (index = YOLO class ID)
DETECTION_CLASSES: list[str] = [
    "player",  # 0 — Cuphead / Mugman player character
    "enemy",  # 1 — All hostile entities (bosses, minions, etc.)
    "projectile",  # 2 — Bullets, fireballs, environmental hazards
    "healthbar",  # 3 — Life counter / health HUD element
]

# YOLO-World text prompts: one prompt per class (same order as DETECTION_CLASSES).
# These are sent as open-vocabulary text queries to YOLO-World.
DETECTION_PROMPTS: list[str] = [
    "cartoon character with cup head",  # 0 — player
    "enemy creature monster boss",  # 1 — enemy
    "bullet projectile shot",  # 2 — projectile
    "health bar life counter hearts HUD",  # 3 — healthbar
]

# ---------------------------------------------------------------------------
# GOLDEN RULES — dataset & inference quality invariants
# ---------------------------------------------------------------------------
# 1. Max 1 player per image. No exceptions found yet.
# 2. Max 1 healthbar per image.
# 3. If unable to accurately find the player label, fall back on pixel
#    color detection (Cuphead = red/white, Mugman = blue/white).
# 4. Always have Claude evaluate <=300 images before training (quality audit).
# ---------------------------------------------------------------------------

# Per-image validation rules: max allowed instances (None = unlimited).
# Used by the dataset build/validation step to flag bad labels.
MAX_PER_IMAGE: dict[str, int] = {
    "player": 1,
    "healthbar": 1,
}

# ---------------------------------------------------------------------------
# Player color validation — Cuphead is red/white, Mugman is blue/white.
# Used to enforce the 1-player-per-image rule: among all "player" detections
# in a single frame, the one with the highest color match keeps the label,
# and the rest are reclassified as "enemy".
#
# Profiled at 640x360 (dataset standard resolution):
#   - Avg player bounding box: ~51x60 px
#   - Cuphead red (r>120, g<100, b<100): present in 100% of crops, mean 4.4%
#   - Mugman blue (r<100, g<150, b>120): present in 100% of crops, mean 2.4%
#   - White (r>200, g>200, b>200): mean 13.4%, strongest shared signal
#   - At this resolution colors bleed, but ratio still discriminates.
# ---------------------------------------------------------------------------

# Pixel thresholds — tuned for 640x360 resolution
_CUPHEAD_RED = {"r_min": 120, "g_max": 100, "b_max": 100}  # red-ish pixels
_MUGMAN_BLUE = {"r_max": 100, "g_max": 150, "b_min": 120}  # blue-ish pixels
_WHITE = {"r_min": 200, "g_min": 200, "b_min": 200}  # white-ish pixels


def player_color_score(pixels):
    """Score how likely a crop of pixels belongs to Cuphead or Mugman.

    Args:
        pixels: numpy array of shape (H, W, 3), dtype uint8, RGB order.

    Returns:
        float score >= 0. Higher = more likely to be the player character.
    """
    import numpy as np

    if pixels.size == 0:
        return 0.0

    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    total = r.size

    # Cuphead: red pixels
    red_ratio = (
        float(
            np.count_nonzero(
                (r > _CUPHEAD_RED["r_min"])
                & (g < _CUPHEAD_RED["g_max"])
                & (b < _CUPHEAD_RED["b_max"])
            )
        )
        / total
    )

    # Mugman: blue pixels
    blue_ratio = (
        float(
            np.count_nonzero(
                (r < _MUGMAN_BLUE["r_max"])
                & (g < _MUGMAN_BLUE["g_max"])
                & (b > _MUGMAN_BLUE["b_min"])
            )
        )
        / total
    )

    # White pixels (shared by both characters)
    white_ratio = (
        float(
            np.count_nonzero(
                (r > _WHITE["r_min"]) & (g > _WHITE["g_min"]) & (b > _WHITE["b_min"])
            )
        )
        / total
    )

    # Score: best character color match + white bonus
    return max(red_ratio, blue_ratio) + white_ratio * 0.3


def enforce_single_player(results, images):
    """Post-process YOLO results: enforce max 1 player per image via color.

    For each image with multiple player detections, scores each by
    player_color_score. The highest-scoring box keeps class 0 (player),
    the rest are changed to class 1 (enemy).

    Args:
        results: list of ultralytics Results objects (mutated in place).
        images: list of numpy arrays (H, W, 3) RGB, or list of file paths.

    Returns:
        dict with stats: {"reclassified": int, "images_fixed": int}
    """
    import numpy as np
    from PIL import Image as PILImage

    stats = {"reclassified": 0, "images_fixed": 0}

    for i, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        cls = boxes.cls.cpu().numpy()
        player_indices = np.where(cls == 0)[0]

        if len(player_indices) <= 1:
            continue  # 0 or 1 player — no conflict

        # Load image pixels
        if isinstance(images[i], str):
            img = np.array(PILImage.open(images[i]).convert("RGB"))
        elif isinstance(images[i], np.ndarray):
            img = images[i]
        else:
            img = np.array(images[i])

        h_img, w_img = img.shape[:2]

        # Score each player box
        scores = []
        xyxy = boxes.xyxy.cpu().numpy()
        for idx in player_indices:
            x1, y1, x2, y2 = xyxy[idx].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop = img[y1:y2, x1:x2]
            scores.append(player_color_score(crop))

        # Best score keeps player label, rest become enemy
        best = player_indices[np.argmax(scores)]
        # cls is a view into boxes.data[:, -1], mutate data directly
        data = r.boxes.data.clone()
        for idx in player_indices:
            if idx != best:
                data[idx, -1] = 1  # reclassify to enemy
                stats["reclassified"] += 1
        r.boxes.data = data

        stats["images_fixed"] += 1

    return stats


# YouTube search queries for dataset building
DATASET_SEARCH_QUERIES: list[str] = [
    "cuphead all bosses no commentary",
    "cuphead run and gun gameplay no commentary",
    "cuphead expert mode all bosses",
    "cuphead DLC bosses no commentary",
    "cuphead walkthrough no commentary",
]


def format_state(state: dict) -> str:
    """Format a Cuphead state dict as a concise one-line summary."""
    parts = []
    if state.get("in_game"):
        parts.append(f"HP:{state.get('hp', 0)}/{state.get('hp_max', 0)}")
        parts.append(
            f"Super:{state.get('super_meter', 0):.0f}"
            f"/{state.get('super_meter_max', 0):.0f}"
        )
        parts.append(f"Deaths:{state.get('deaths', 0)}")
        parts.append(f"Time:{state.get('level_time', 0):.1f}s")
        hits = state.get("num_times_hit", 0)
        parries = state.get("num_parries", 0)
        if hits or parries:
            parts.append(f"Hit:{hits} Par:{parries}")
        if state.get("level_ending"):
            parts.append("ENDING")
        if state.get("level_won"):
            parts.append("WON")
        mode = state.get("level_mode", -1)
        parts.append(f"Mode:{MODE_NAMES.get(mode, '?')}")
    else:
        parts.append("Not in game")
    scene = state.get("scene_name", "")
    if scene:
        parts.append(f"Scene:{scene}")
    if state.get("is_loading"):
        parts.append("LOADING")
    return " | ".join(parts)
