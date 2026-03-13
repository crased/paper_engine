"""
Self-Learning Pipeline — orchestrates the full data generation + training loop.

Chains together:
  1. Memory bot plays Cuphead → GameplayRecorder captures frames + state
  2. Batch annotator sends frames to Gemini → generates YOLO labels
  3. YOLO model trains on expanded dataset

Each step can also be run independently via their own CLIs.

Usage:
    # Full pipeline (bot plays, annotates, trains):
    python -m pipeline.self_learning --play-duration 120 --annotate --train

    # Annotate an existing session, then train:
    python -m pipeline.self_learning --session recordings/sessions/Cuphead_... --annotate --train

    # Just annotate the most recent session:
    python -m pipeline.self_learning --latest --annotate

    # Dry run to see what would happen:
    python -m pipeline.self_learning --latest --annotate --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_session(sessions_root: Optional[Path] = None) -> Optional[Path]:
    """Find the most recent recording session directory."""
    if sessions_root is None:
        sessions_root = _PROJECT_ROOT / "recordings" / "sessions"

    if not sessions_root.exists():
        return None

    sessions = [
        d
        for d in sessions_root.iterdir()
        if d.is_dir() and (d / "session.json").exists()
    ]
    if not sessions:
        return None

    return max(sessions, key=lambda d: d.stat().st_mtime)


def count_session_frames(session_dir: Path) -> int:
    """Count PNG frames in a session directory."""
    return len(list(session_dir.glob("frame_*.png")))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_play(
    duration: float = 120.0,
    record_interval: float = 0.5,
    enable_play: bool = True,
    monitor: str = "DP-1",
) -> Optional[Path]:
    """
    Step 1: Run memory bot to play the game and record frames.
    Returns the session directory path.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Playing game with memory bot")
    logger.info(
        "  Duration: %.0fs, Interval: %.1fs, Play: %s",
        duration,
        record_interval,
        enable_play,
    )
    logger.info("=" * 60)

    from bot_scripts.cuphead_bot import CupheadBot

    bot = CupheadBot(
        enable_play=enable_play,
        enable_yolo=False,  # Memory-only during training data generation
        enable_record=True,
        record_interval=record_interval,
        target_monitor=monitor,
    )

    # We need to modify the bot to auto-stop after duration
    # For now, run it in-process with a timer
    import threading

    timer = None

    if duration > 0:

        def _stop_bot():
            logger.info("Duration limit reached (%.0fs) — stopping bot", duration)
            bot.running = False

        timer = threading.Timer(duration, _stop_bot)
        timer.daemon = True
        timer.start()

    try:
        bot.run()
    finally:
        if timer:
            timer.cancel()

    # Find the session that was just created
    session = find_latest_session()
    if session:
        frames = count_session_frames(session)
        logger.info("Play session complete: %d frames in %s", frames, session)
    else:
        logger.warning("No session directory found after play")

    return session


def step_annotate(
    session_dir: Path,
    rpm: int = 10,
    dry_run: bool = False,
    max_frames: int = 0,
) -> bool:
    """
    Step 2: Annotate recorded session frames via Gemini.
    Returns True if annotation succeeded.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Annotating session with Gemini")
    logger.info("  Session: %s", session_dir)
    logger.info("  RPM: %d, Dry run: %s", rpm, dry_run)
    logger.info("=" * 60)

    from pipeline.batch_annotator import annotate_session

    stats = annotate_session(
        session_dir=session_dir,
        rpm=rpm,
        dry_run=dry_run,
        max_frames=max_frames,
    )

    print(f"\n{stats.summary()}")

    if stats.errors > stats.annotated:
        logger.error(
            "Too many errors during annotation (%d errors vs %d annotated)",
            stats.errors,
            stats.annotated,
        )
        return False

    return stats.annotated > 0 or stats.skipped_existing > 0


def step_train(
    dataset_path: Optional[Path] = None,
) -> bool:
    """
    Step 3: Train YOLO model on the dataset (now including new annotations).
    Returns True if training succeeded.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Training YOLO model")
    logger.info("=" * 60)

    if dataset_path is None:
        dataset_path = _PROJECT_ROOT / "yolo_dataset"

    yaml_file = dataset_path / "dataset.yaml"
    if not yaml_file.exists():
        logger.error("Dataset YAML not found: %s", yaml_file)
        return False

    # Count training images
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    if train_images.exists():
        n_images = len(
            list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
        )
        n_labels = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0
        logger.info("Dataset: %d images, %d labels", n_images, n_labels)

    try:
        from pipeline.training_model import train_model
        from conf.config_parser import training_conf

        model = train_model(
            dataset_yaml=yaml_file,
            epochs=training_conf.EPOCHS,
            img_size=training_conf.IMG_SIZE,
            batch_size=training_conf.BATCH_SIZE,
            model_name=training_conf.MODEL_NAME,
        )
        logger.info("Training complete")
        return True
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-learning pipeline: play → annotate → train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full loop (play 2 min, annotate, train):
    python -m pipeline.self_learning --play-duration 120 --annotate --train

  Annotate the latest session:
    python -m pipeline.self_learning --latest --annotate

  Annotate a specific session:
    python -m pipeline.self_learning --session path/to/session --annotate

  Record only (no bot input), then annotate:
    python -m pipeline.self_learning --play-duration 60 --no-play --annotate

  Just train on existing data:
    python -m pipeline.self_learning --train
""",
    )

    # Step 1: Play
    play_group = parser.add_argument_group("Play (Step 1)")
    play_group.add_argument(
        "--play-duration",
        type=float,
        default=0,
        help="Play for N seconds (0 = skip play step)",
    )
    play_group.add_argument(
        "--no-play", action="store_true", help="Record without bot input"
    )
    play_group.add_argument(
        "--record-interval",
        type=float,
        default=0.5,
        help="Capture interval in seconds (default: 0.5)",
    )
    play_group.add_argument(
        "--monitor", type=str, default="DP-1", help="Target monitor (default: DP-1)"
    )

    # Step 2: Annotate
    annot_group = parser.add_argument_group("Annotate (Step 2)")
    annot_group.add_argument(
        "--annotate", action="store_true", help="Run Gemini annotation on session"
    )
    annot_group.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to session directory to annotate",
    )
    annot_group.add_argument(
        "--latest", action="store_true", help="Use the most recent session"
    )
    annot_group.add_argument(
        "--rpm", type=int, default=10, help="Gemini requests per minute (default: 10)"
    )
    annot_group.add_argument(
        "--max-frames", type=int, default=0, help="Limit annotation to N frames"
    )
    annot_group.add_argument(
        "--dry-run", action="store_true", help="Preview annotation without API calls"
    )

    # Step 3: Train
    train_group = parser.add_argument_group("Train (Step 3)")
    train_group.add_argument(
        "--train", action="store_true", help="Train YOLO model after annotation"
    )

    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine which steps to run
    do_play = args.play_duration > 0
    do_annotate = args.annotate
    do_train = args.train

    if not do_play and not do_annotate and not do_train:
        parser.print_help()
        print(
            "\nError: specify at least one step (--play-duration, --annotate, --train)"
        )
        sys.exit(1)

    session_dir = None
    start_time = time.time()

    # Step 1: Play
    if do_play:
        session_dir = step_play(
            duration=args.play_duration,
            record_interval=args.record_interval,
            enable_play=not args.no_play,
            monitor=args.monitor,
        )
        if not session_dir:
            logger.error("Play step produced no session — aborting")
            sys.exit(1)

    # Determine session for annotation
    if do_annotate:
        if args.session:
            session_dir = Path(args.session)
        elif args.latest or session_dir is None:
            session_dir = find_latest_session()
            if session_dir:
                logger.info("Using latest session: %s", session_dir)

        if not session_dir or not session_dir.exists():
            logger.error("No session directory available for annotation")
            if not do_train:
                sys.exit(1)
        else:
            # Step 2: Annotate
            ok = step_annotate(
                session_dir=session_dir,
                rpm=args.rpm,
                dry_run=args.dry_run,
                max_frames=args.max_frames,
            )
            if not ok:
                logger.warning("Annotation step had issues")

    # Step 3: Train
    if do_train:
        ok = step_train()
        if not ok:
            logger.error("Training step failed")
            sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Pipeline complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
