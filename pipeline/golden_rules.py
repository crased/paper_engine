"""
Golden Rules — persistent annotation insights that improve over time.

Rules are universal truths about how objects look and behave in the game.
They are derived from human review (descriptions, corrections, patterns)
and injected into LLM annotation prompts to improve accuracy.

Three rule types:
  - class_rules:   Per-class appearance/behavior (e.g., "player is ~80x100px")
  - scene_rules:   Per-scene-type patterns (e.g., "boss fights have 1 large enemy")
  - quality_rules: Annotation standards (e.g., "boxes must be tight, no padding")

Storage: conf/golden_rules.json
Injected into: batch_annotator prompts, describe_annotator prompts

Usage:
    from pipeline.golden_rules import RuleStore

    store = RuleStore()
    store.add("class", "player", "Always blue/white, ~80x100px at 1080p, cup-shaped head")
    store.add("class", "projectile", "Never bigger than the player, usually pink or orange")
    store.add("scene", "world_map", "Characters on the map are NPCs (enemy class), not hostile")
    store.add("quality", "tightness", "Boxes must tightly enclose the object, no extra padding")
    store.save()

    # For injection into prompts:
    prompt_block = store.to_prompt_block()

CLI:
    python -m pipeline.golden_rules --list
    python -m pipeline.golden_rules --add class player "Always blue/white cup character"
    python -m pipeline.golden_rules --remove class player 0
    python -m pipeline.golden_rules --from-feedback reports/review_feedback/review_*.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

_DEFAULT_PATH = _project_root / "conf" / "golden_rules.json"


@dataclass
class Rule:
    """A single annotation insight."""

    text: str  # the rule itself
    source: str = ""  # where it came from ("manual", "review_20260325", etc.)
    created: str = ""  # ISO timestamp
    confidence: float = 1.0  # 0-1, how certain (manual=1.0, auto-derived=lower)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "created": self.created,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Rule:
        return cls(
            text=d["text"],
            source=d.get("source", ""),
            created=d.get("created", ""),
            confidence=d.get("confidence", 1.0),
        )


class RuleStore:
    """Persistent store for golden rules."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else _DEFAULT_PATH
        # Structure: {type: {key: [Rule, ...]}}
        # type = "class", "scene", "quality"
        # key = class name, scene type, or quality aspect
        self._rules: Dict[str, Dict[str, List[Rule]]] = {
            "class": {},
            "scene": {},
            "quality": {},
        }
        self.load()

    def load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for rule_type in ("class", "scene", "quality"):
                section = data.get(rule_type, {})
                for key, rules_list in section.items():
                    self._rules[rule_type][key] = [
                        Rule.from_dict(r) for r in rules_list
                    ]
        except Exception as exc:
            logger.warning("Failed to load golden rules: %s", exc)

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for rule_type, section in self._rules.items():
            data[rule_type] = {}
            for key, rules in section.items():
                data[rule_type][key] = [r.to_dict() for r in rules]
        self._path.write_text(json.dumps(data, indent=2))

    def add(self, rule_type: str, key: str, text: str, source: str = "manual"):
        """Add a rule. Deduplicates by exact text match."""
        if rule_type not in self._rules:
            raise ValueError(f"Unknown rule type: {rule_type}")
        if key not in self._rules[rule_type]:
            self._rules[rule_type][key] = []

        # Deduplicate
        for existing in self._rules[rule_type][key]:
            if existing.text == text:
                return

        self._rules[rule_type][key].append(Rule(
            text=text,
            source=source,
            created=datetime.now().isoformat(),
        ))

    def remove(self, rule_type: str, key: str, index: int):
        """Remove a rule by index."""
        rules = self._rules.get(rule_type, {}).get(key, [])
        if 0 <= index < len(rules):
            rules.pop(index)

    def get(self, rule_type: str, key: str) -> List[Rule]:
        return self._rules.get(rule_type, {}).get(key, [])

    def all_rules(self) -> Dict[str, Dict[str, List[Rule]]]:
        return self._rules

    def to_prompt_block(self) -> str:
        """Format all rules as a text block for injection into LLM prompts."""
        lines = []

        # Class rules
        class_rules = self._rules.get("class", {})
        if class_rules:
            lines.append("ANNOTATION RULES (per class):")
            for class_name, rules in sorted(class_rules.items()):
                for r in rules:
                    lines.append(f"  - {class_name}: {r.text}")

        # Scene rules
        scene_rules = self._rules.get("scene", {})
        if scene_rules:
            lines.append("\nSCENE RULES:")
            for scene_type, rules in sorted(scene_rules.items()):
                for r in rules:
                    lines.append(f"  - {scene_type}: {r.text}")

        # Quality rules
        quality_rules = self._rules.get("quality", {})
        if quality_rules:
            lines.append("\nQUALITY STANDARDS:")
            for aspect, rules in sorted(quality_rules.items()):
                for r in rules:
                    lines.append(f"  - {r.text}")

        return "\n".join(lines) if lines else ""

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        for rule_type, section in self._rules.items():
            count = sum(len(rules) for rules in section.values())
            if count:
                lines.append(f"\n=== {rule_type.upper()} RULES ({count}) ===")
                for key, rules in sorted(section.items()):
                    for i, r in enumerate(rules):
                        src = f"  [{r.source}]" if r.source else ""
                        lines.append(f"  [{i}] {key}: {r.text}{src}")
        return "\n".join(lines) if lines else "(no rules yet)"


def extract_rules_from_feedback(feedback_paths: List[str]) -> List[dict]:
    """Analyze review feedback sessions to suggest rules.

    Looks for patterns in corrections:
    - Classes that get frequently reclassified → class rule
    - Boxes that consistently get resized in one direction → size rule
    - Images where objects were added/removed → missing object patterns

    Returns list of suggested rules for human approval.
    """
    from pipeline.review_feedback import ReviewSession

    suggestions = []
    class_corrections = {}  # (from_class, to_class) → count
    size_adjustments = {}  # class → list of (dx, dy) adjustments
    added_classes = {}  # class → count of times added during review
    removed_classes = {}  # class → count of times removed

    for path in feedback_paths:
        try:
            session = ReviewSession.load(Path(path))
        except Exception:
            continue

        for review in session.reviews:
            if not review.corrections_made:
                continue

            gen_by_class = {}
            for d in review.generated_detections:
                cid = d.get("class_id")
                gen_by_class.setdefault(cid, []).append(d)

            final_by_class = {}
            for d in review.final_detections:
                cid = d.get("class_id")
                final_by_class.setdefault(cid, []).append(d)

            # Track classes added (in final but not generated)
            for cid in final_by_class:
                if cid not in gen_by_class:
                    name = final_by_class[cid][0].get("class_name", str(cid))
                    added_classes[name] = added_classes.get(name, 0) + len(final_by_class[cid])

            # Track classes removed (in generated but not final)
            for cid in gen_by_class:
                if cid not in final_by_class:
                    name = gen_by_class[cid][0].get("class_name", str(cid))
                    removed_classes[name] = removed_classes.get(name, 0) + len(gen_by_class[cid])

    # Generate suggestions from patterns
    for class_name, count in added_classes.items():
        if count >= 3:
            suggestions.append({
                "type": "class",
                "key": class_name,
                "text": f"LLM frequently misses {class_name} — annotators added it {count} times",
                "confidence": min(count / 10, 0.9),
            })

    for class_name, count in removed_classes.items():
        if count >= 3:
            suggestions.append({
                "type": "class",
                "key": class_name,
                "text": f"LLM over-detects {class_name} — annotators removed it {count} times",
                "confidence": min(count / 10, 0.9),
            })

    return suggestions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Manage golden annotation rules")
    parser.add_argument("--list", action="store_true", help="List all rules")
    parser.add_argument("--add", nargs=3, metavar=("TYPE", "KEY", "TEXT"),
                        help="Add a rule (type: class/scene/quality)")
    parser.add_argument("--remove", nargs=3, metavar=("TYPE", "KEY", "INDEX"),
                        help="Remove a rule by index")
    parser.add_argument("--prompt", action="store_true",
                        help="Print the prompt block that gets injected into LLM calls")
    parser.add_argument("--from-feedback", nargs="+", metavar="PATH",
                        help="Suggest rules from review feedback sessions")
    args = parser.parse_args()

    store = RuleStore()

    if args.add:
        rule_type, key, text = args.add
        store.add(rule_type, key, text)
        store.save()
        print(f"Added {rule_type} rule for '{key}'")

    elif args.remove:
        rule_type, key, index = args.remove
        store.remove(rule_type, key, int(index))
        store.save()
        print(f"Removed rule [{index}] from {rule_type}/{key}")

    elif args.prompt:
        block = store.to_prompt_block()
        print(block if block else "(no rules — prompt block is empty)")

    elif args.from_feedback:
        suggestions = extract_rules_from_feedback(args.from_feedback)
        if not suggestions:
            print("No patterns found in feedback.")
        else:
            print(f"\n{len(suggestions)} suggested rules:\n")
            for i, s in enumerate(suggestions):
                print(f"  [{i}] ({s['type']}/{s['key']}) {s['text']}")
            print(f"\nUse --add to accept any of these.")

    else:
        print(store.summary())


if __name__ == "__main__":
    main()
