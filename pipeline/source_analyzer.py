"""
Game Source Code Analyzer — static analysis of decompiled game code.

Reads decompiled source files (C#, C++, GDScript, etc.) from a game directory
and extracts structural insights: classes, fields, methods, inheritance,
constants, and patterns. Results feed into the directory-based game report
as Source 9.

Adapted from the Checker/Diagnostic pattern in static_code_analyzer/.
Since game source is not Python, uses regex-based structure extraction
instead of AST walking, then optionally sends to LLM for semantic analysis.

Supports:
  - Unity/Mono (decompiled C# from dnSpy/ILSpy/dotPeek)
  - Unreal Engine (C++ headers, Blueprint references)
  - Godot (GDScript .gd files)
  - Generic (any text source with class/function patterns)

Usage:
    from pipeline.source_analyzer import analyze_game_source

    result = analyze_game_source(Path("/path/to/game"))
    print(result.summary())
    prompt_block = result.to_prompt_block()  # for LLM injection

CLI:
    python -m pipeline.source_analyzer /path/to/game
    python -m pipeline.source_analyzer /path/to/game --json
    python -m pipeline.source_analyzer /path/to/game --decompiled /path/to/decompiled
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Diagnostic — same pattern as static_code_analyzer
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """A single finding from source analysis."""

    file: str
    line: int
    col: int
    rule_id: str
    message: str
    severity: str = "info"  # info, warning, insight

    def __str__(self):
        return f"{self.file}:{self.line}:{self.col}: {self.rule_id} {self.message}"


# ---------------------------------------------------------------------------
# Extracted structures
# ---------------------------------------------------------------------------


@dataclass
class ClassInfo:
    """A class/struct found in game source."""

    name: str
    file: str
    line: int
    base_classes: List[str] = field(default_factory=list)
    fields: List[Dict[str, str]] = field(default_factory=list)  # [{name, type, access}]
    methods: List[Dict[str, str]] = field(default_factory=list)  # [{name, return_type, access}]
    is_monobehaviour: bool = False  # Unity-specific
    is_component: bool = False


@dataclass
class SourceAnalysis:
    """Complete analysis result for a game's source code."""

    game_dir: str
    source_dir: str  # where decompiled source was found/provided
    engine: str = ""  # unity, unreal, godot, generic
    language: str = ""  # csharp, cpp, gdscript

    classes: List[ClassInfo] = field(default_factory=list)
    diagnostics: List[Diagnostic] = field(default_factory=list)
    constants: Dict[str, str] = field(default_factory=dict)  # name → value
    enums: Dict[str, List[str]] = field(default_factory=dict)  # enum_name → [values]
    inheritance_tree: Dict[str, List[str]] = field(default_factory=dict)  # base → [children]

    # Stats
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_methods: int = 0

    def summary(self) -> str:
        lines = [
            f"Source Analysis: {self.engine}/{self.language}",
            f"  Files: {self.total_files}, Lines: {self.total_lines}",
            f"  Classes: {self.total_classes}, Methods: {self.total_methods}",
            f"  Constants: {len(self.constants)}, Enums: {len(self.enums)}",
            f"  Diagnostics: {len(self.diagnostics)}",
        ]

        if self.classes:
            lines.append(f"\n  Key classes:")
            # Show MonoBehaviour subclasses first (game logic)
            mono = [c for c in self.classes if c.is_monobehaviour]
            other = [c for c in self.classes if not c.is_monobehaviour]
            for c in (mono[:10] + other[:5]):
                bases = f" : {', '.join(c.base_classes)}" if c.base_classes else ""
                lines.append(f"    {c.name}{bases}  ({len(c.fields)}F, {len(c.methods)}M)")

        if self.diagnostics:
            lines.append(f"\n  Findings:")
            for d in self.diagnostics[:15]:
                lines.append(f"    {d}")

        return "\n".join(lines)

    def to_prompt_block(self) -> str:
        """Format for injection into LLM game report prompts."""
        lines = [
            f"SOURCE CODE ANALYSIS ({self.engine}, {self.language}):",
            f"  {self.total_files} files, {self.total_classes} classes, {self.total_methods} methods",
        ]

        if self.classes:
            lines.append("\n  CLASS HIERARCHY:")
            mono = [c for c in self.classes if c.is_monobehaviour]
            for c in mono[:20]:
                bases = f" : {', '.join(c.base_classes)}" if c.base_classes else ""
                fields_str = ", ".join(f.get("name", "") for f in c.fields[:5])
                if len(c.fields) > 5:
                    fields_str += f" (+{len(c.fields)-5} more)"
                lines.append(f"    {c.name}{bases}")
                if fields_str:
                    lines.append(f"      fields: {fields_str}")

        if self.enums:
            lines.append("\n  ENUMS:")
            for name, values in list(self.enums.items())[:10]:
                lines.append(f"    {name}: {', '.join(values[:8])}")

        if self.constants:
            lines.append("\n  CONSTANTS:")
            for name, val in list(self.constants.items())[:15]:
                lines.append(f"    {name} = {val}")

        if self.diagnostics:
            lines.append("\n  FINDINGS:")
            for d in self.diagnostics[:10]:
                lines.append(f"    [{d.severity}] {d.rule_id}: {d.message}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "game_dir": self.game_dir,
            "source_dir": self.source_dir,
            "engine": self.engine,
            "language": self.language,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_classes": self.total_classes,
            "total_methods": self.total_methods,
            "classes": [
                {
                    "name": c.name,
                    "file": c.file,
                    "base_classes": c.base_classes,
                    "fields": c.fields,
                    "methods": c.methods,
                    "is_monobehaviour": c.is_monobehaviour,
                }
                for c in self.classes
            ],
            "constants": self.constants,
            "enums": self.enums,
            "diagnostics": [str(d) for d in self.diagnostics],
        }


# ---------------------------------------------------------------------------
# Checker base — regex-based, language-agnostic
# ---------------------------------------------------------------------------


class Checker:
    """Base class for source code checkers. Subclasses implement check()."""

    def __init__(self, filename: str):
        self.filename = filename
        self.diagnostics: List[Diagnostic] = []

    def add_diagnostic(self, line: int, rule_id: str, message: str,
                       severity: str = "info", col: int = 0):
        self.diagnostics.append(Diagnostic(
            file=self.filename, line=line, col=col,
            rule_id=rule_id, message=message, severity=severity,
        ))

    def check(self, source: str, analysis: SourceAnalysis):
        """Override in subclass. Analyze source text and populate diagnostics."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# C# Checkers (Unity/Mono decompiled code)
# ---------------------------------------------------------------------------


class CSharpClassExtractor(Checker):
    """Extract class definitions, fields, methods, and inheritance from C#."""

    # class ClassName : BaseClass, IInterface
    _CLASS_RE = re.compile(
        r'^\s*(?:public|private|protected|internal|static|abstract|sealed|\s)*'
        r'class\s+(\w+)'
        r'(?:\s*:\s*([\w\s,.<>]+))?',
        re.MULTILINE,
    )

    # public int fieldName;  or  private float _speed = 5f;
    _FIELD_RE = re.compile(
        r'^\s*(public|private|protected|internal)\s+'
        r'(?:static\s+|readonly\s+|const\s+|volatile\s+)*'
        r'([\w<>\[\],\s\.]+?)\s+(\w+)\s*[;=]',
        re.MULTILINE,
    )

    # public void MethodName(...)  or  private int GetValue()
    _METHOD_RE = re.compile(
        r'^\s*(public|private|protected|internal|override|virtual|static|\s)*'
        r'([\w<>\[\]\.]+)\s+(\w+)\s*\(',
        re.MULTILINE,
    )

    # enum EnumName { Value1, Value2 }
    _ENUM_RE = re.compile(
        r'^\s*(?:public|private|protected|internal|\s)*enum\s+(\w+)\s*\{([^}]*)\}',
        re.MULTILINE | re.DOTALL,
    )

    # const int NAME = value;  or  static readonly TYPE NAME = value;
    _CONST_RE = re.compile(
        r'^\s*(?:public|private|protected|internal)\s+'
        r'(?:static\s+readonly|const)\s+'
        r'[\w<>\[\]\.]+\s+(\w+)\s*=\s*(.+?)\s*;',
        re.MULTILINE,
    )

    _MONO_BASES = {
        "MonoBehaviour", "ScriptableObject", "NetworkBehaviour",
        "StateMachineBehaviour", "Component",
    }

    def check(self, source: str, analysis: SourceAnalysis):
        lines = source.split("\n")

        # Extract classes
        for match in self._CLASS_RE.finditer(source):
            name = match.group(1)
            bases_raw = match.group(2) or ""
            bases = [b.strip() for b in bases_raw.split(",") if b.strip()]
            line_num = source[:match.start()].count("\n") + 1

            info = ClassInfo(
                name=name,
                file=self.filename,
                line=line_num,
                base_classes=bases,
            )

            # Check if it's a Unity component
            for base in bases:
                if base in self._MONO_BASES or "Behaviour" in base:
                    info.is_monobehaviour = True
                    info.is_component = True
                    break

            analysis.classes.append(info)
            analysis.total_classes += 1

            # Build inheritance tree
            for base in bases:
                analysis.inheritance_tree.setdefault(base, []).append(name)

        # Extract fields (attach to most recent class)
        current_class = None
        for match in self._FIELD_RE.finditer(source):
            access = match.group(1)
            ftype = match.group(2).strip()
            fname = match.group(3)
            line_num = source[:match.start()].count("\n") + 1

            # Find which class this belongs to
            for cls in reversed(analysis.classes):
                if cls.file == self.filename and cls.line <= line_num:
                    cls.fields.append({"name": fname, "type": ftype, "access": access})
                    break

        # Extract methods
        for match in self._METHOD_RE.finditer(source):
            access_and_modifiers = (match.group(1) or "").strip()
            ret_type = match.group(2).strip()
            mname = match.group(3)
            line_num = source[:match.start()].count("\n") + 1

            # Skip constructors and common non-methods
            if ret_type in ("class", "struct", "enum", "namespace", "if", "for",
                            "while", "switch", "return", "new", "using"):
                continue

            access = "private"
            for a in ("public", "private", "protected", "internal"):
                if a in access_and_modifiers:
                    access = a
                    break

            analysis.total_methods += 1

            for cls in reversed(analysis.classes):
                if cls.file == self.filename and cls.line <= line_num:
                    cls.methods.append({
                        "name": mname, "return_type": ret_type, "access": access,
                    })
                    break

        # Extract enums
        for match in self._ENUM_RE.finditer(source):
            enum_name = match.group(1)
            values_raw = match.group(2)
            values = [v.strip().split("=")[0].strip()
                      for v in values_raw.split(",") if v.strip()]
            analysis.enums[enum_name] = values

        # Extract constants
        for match in self._CONST_RE.finditer(source):
            const_name = match.group(1)
            const_val = match.group(2).strip()
            analysis.constants[const_name] = const_val


class CSharpPatternChecker(Checker):
    """Detect common patterns and anti-patterns in C# game code."""

    def check(self, source: str, analysis: SourceAnalysis):
        lines = source.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Singleton pattern
            if re.search(r'static\s+\w+\s+[Ii]nstance\s*[{;=]', stripped):
                self.add_diagnostic(i, "GS001", f"Singleton pattern detected", "insight")

            # Update/FixedUpdate (Unity hot path)
            if re.match(r'\s*(?:void|private\s+void|public\s+void)\s+(?:Update|FixedUpdate|LateUpdate)\s*\(', stripped):
                self.add_diagnostic(i, "GS002", "Unity update loop — performance-sensitive method", "insight")

            # GetComponent in loop (performance issue)
            if "GetComponent" in stripped and any(
                kw in stripped for kw in ("Update", "FixedUpdate", "for", "while", "foreach")
            ):
                self.add_diagnostic(i, "GS003",
                    "GetComponent in hot path — should cache in Awake/Start", "warning")

            # Magic numbers in game logic
            if re.search(r'(?:hp|health|damage|speed|force)\s*[=<>!]+\s*\d+', stripped, re.IGNORECASE):
                self.add_diagnostic(i, "GS004",
                    "Hardcoded game value — should be configurable", "info")

            # Coroutine allocation
            if "new WaitForSeconds" in stripped:
                self.add_diagnostic(i, "GS005",
                    "WaitForSeconds allocation — cache for GC reduction", "info")

            # Find/FindObjectOfType (expensive)
            if re.search(r'(?:Find|FindObjectOfType|FindGameObjectWithTag)\s*[<(]', stripped):
                self.add_diagnostic(i, "GS006",
                    "Runtime Find call — expensive, prefer cached references", "warning")

            # State machine patterns
            if re.search(r'enum\s+\w*[Ss]tate', stripped):
                self.add_diagnostic(i, "GS007", "State enum detected", "insight")

            # Serialized fields (designer-tunable values)
            if "[SerializeField]" in stripped or "[Header(" in stripped:
                self.add_diagnostic(i, "GS008", "Serialized field — designer-tunable value", "insight")


class GDScriptExtractor(Checker):
    """Extract structure from Godot GDScript files."""

    _CLASS_RE = re.compile(r'^class_name\s+(\w+)', re.MULTILINE)
    _EXTENDS_RE = re.compile(r'^extends\s+(\w+)', re.MULTILINE)
    _FUNC_RE = re.compile(r'^func\s+(\w+)\s*\(', re.MULTILINE)
    _VAR_RE = re.compile(
        r'^(?:export\s+)?(?:onready\s+)?var\s+(\w+)(?:\s*:\s*(\w+))?',
        re.MULTILINE,
    )
    _SIGNAL_RE = re.compile(r'^signal\s+(\w+)', re.MULTILINE)

    def check(self, source: str, analysis: SourceAnalysis):
        # Class name
        class_match = self._CLASS_RE.search(source)
        extends_match = self._EXTENDS_RE.search(source)

        name = class_match.group(1) if class_match else Path(self.filename).stem
        bases = [extends_match.group(1)] if extends_match else []

        info = ClassInfo(
            name=name, file=self.filename, line=1,
            base_classes=bases,
        )

        # Functions
        for match in self._FUNC_RE.finditer(source):
            fname = match.group(1)
            line_num = source[:match.start()].count("\n") + 1
            info.methods.append({"name": fname, "return_type": "", "access": "public"})
            analysis.total_methods += 1

        # Variables
        for match in self._VAR_RE.finditer(source):
            vname = match.group(1)
            vtype = match.group(2) or ""
            info.fields.append({"name": vname, "type": vtype, "access": "public"})

        # Signals
        for match in self._SIGNAL_RE.finditer(source):
            info.fields.append({"name": match.group(1), "type": "signal", "access": "public"})

        analysis.classes.append(info)
        analysis.total_classes += 1

        if bases:
            for base in bases:
                analysis.inheritance_tree.setdefault(base, []).append(name)


# ---------------------------------------------------------------------------
# Source file discovery
# ---------------------------------------------------------------------------

# Patterns for finding decompiled/source code
_SOURCE_PATTERNS = {
    "csharp": [
        "**/*.cs",
        "**/Assembly-CSharp/**/*.cs",
        "**/Managed/decompiled/**/*.cs",
    ],
    "cpp": [
        "**/Source/**/*.h",
        "**/Source/**/*.cpp",
    ],
    "gdscript": [
        "**/*.gd",
    ],
}

# Max files to analyze per language (to avoid scanning 10k files)
_MAX_FILES = 200
_MAX_FILE_SIZE = 100_000  # 100KB per file


def _find_source_files(
    game_dir: Path,
    decompiled_dir: Optional[Path] = None,
) -> Tuple[str, List[Path]]:
    """Discover source files and determine language.

    Checks decompiled_dir first (if provided), then game_dir.
    Returns (language, [file_paths]).
    """
    search_dirs = []
    if decompiled_dir and decompiled_dir.is_dir():
        search_dirs.append(decompiled_dir)
    search_dirs.append(game_dir)

    for lang, patterns in _SOURCE_PATTERNS.items():
        files = []
        for search_dir in search_dirs:
            for pattern in patterns:
                found = sorted(search_dir.glob(pattern))
                files.extend(f for f in found if f.is_file() and f.stat().st_size < _MAX_FILE_SIZE)
                if len(files) >= _MAX_FILES:
                    break
            if files:
                break
        if files:
            return lang, files[:_MAX_FILES]

    return "", []


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_game_source(
    game_dir: Path,
    decompiled_dir: Optional[Path] = None,
    engine_hint: str = "",
    log_fn=None,
) -> SourceAnalysis:
    """Analyze game source code from a game directory.

    Args:
        game_dir: Path to game install.
        decompiled_dir: Optional path to decompiled source (e.g., from dnSpy).
        engine_hint: Engine name hint from detect_engine() (e.g., "unity").
        log_fn: Optional progress callback.

    Returns:
        SourceAnalysis with extracted classes, fields, patterns, diagnostics.
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)

    analysis = SourceAnalysis(
        game_dir=str(game_dir),
        source_dir=str(decompiled_dir or game_dir),
        engine=engine_hint,
    )

    # Find source files
    _log("  Searching for source files...\n")
    language, files = _find_source_files(game_dir, decompiled_dir)

    if not files:
        _log("  No source files found.\n")
        return analysis

    analysis.language = language
    analysis.total_files = len(files)
    _log(f"  Found {len(files)} {language} files\n")

    # Select checkers based on language
    if language == "csharp":
        checker_classes = [CSharpClassExtractor, CSharpPatternChecker]
    elif language == "gdscript":
        checker_classes = [GDScriptExtractor]
    else:
        checker_classes = []

    # Run analysis
    for fpath in files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            analysis.total_lines += source.count("\n") + 1
            rel = str(fpath.relative_to(decompiled_dir or game_dir))

            for checker_cls in checker_classes:
                checker = checker_cls(rel)
                checker.check(source, analysis)
                analysis.diagnostics.extend(checker.diagnostics)

        except Exception as exc:
            logger.debug("Error analyzing %s: %s", fpath, exc)

    # Sort diagnostics
    analysis.diagnostics.sort(key=lambda d: (d.file, d.line))

    _log(f"  Extracted {analysis.total_classes} classes, "
         f"{analysis.total_methods} methods, "
         f"{len(analysis.diagnostics)} findings\n")

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze game source code")
    parser.add_argument("game_dir", help="Path to game directory")
    parser.add_argument("--decompiled", default=None, help="Path to decompiled source")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--prompt", action="store_true", help="Output as LLM prompt block")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = analyze_game_source(
        game_dir=Path(args.game_dir),
        decompiled_dir=Path(args.decompiled) if args.decompiled else None,
        log_fn=print,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.prompt:
        print(result.to_prompt_block())
    else:
        print(result.summary())


if __name__ == "__main__":
    main()
