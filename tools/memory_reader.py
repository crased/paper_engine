"""
Low-level Linux memory reading for game processes running under Wine.

Uses process_vm_readv (via ctypes) for fast cross-process reads without
ptrace attach. Falls back to /proc/<pid>/mem if needed.

Key concepts for someone new to memory reading:
- Every process has a virtual address space (like a giant byte array)
- /proc/<pid>/maps shows which regions of that array are mapped (code, heap, etc.)
- process_vm_readv lets us read bytes from another process's address space
- A "pointer" is just a number that's an index into that array
- "Pointer chains" follow address → read value at address → that value is
  another address → read there → etc. until you reach the actual data
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# libc bindings
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class _iovec(ctypes.Structure):
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


# ssize_t process_vm_readv(pid_t pid,
#     const struct iovec *local_iov, unsigned long liovcnt,
#     const struct iovec *remote_iov, unsigned long riovcnt,
#     unsigned long flags);
_process_vm_readv = _libc.process_vm_readv
_process_vm_readv.restype = ctypes.c_ssize_t
_process_vm_readv.argtypes = [
    ctypes.c_int,  # pid
    ctypes.POINTER(_iovec),  # local_iov
    ctypes.c_ulong,  # liovcnt
    ctypes.POINTER(_iovec),  # remote_iov
    ctypes.c_ulong,  # riovcnt
    ctypes.c_ulong,  # flags
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MemoryRegion:
    """A mapped region from /proc/<pid>/maps."""

    start: int
    end: int
    perms: str  # e.g. "r-xp"
    offset: int
    device: str
    inode: int
    pathname: str

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def readable(self) -> bool:
        return "r" in self.perms

    @property
    def writable(self) -> bool:
        return "w" in self.perms

    @property
    def executable(self) -> bool:
        return "x" in self.perms


# Regex for /proc/<pid>/maps lines:
# 7f1234000000-7f1234001000 r-xp 00000000 08:01 12345  /path/to/lib.so
_MAPS_RE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+)\s+"  # address range
    r"(\S+)\s+"  # permissions
    r"([0-9a-f]+)\s+"  # offset
    r"(\S+)\s+"  # device
    r"(\d+)\s*"  # inode
    r"(.*)$",  # pathname (may be empty)
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Game launcher (spawns as child process for ptrace access)
# ---------------------------------------------------------------------------

# Module-level reference to the launched game subprocess.
# Kept alive so the child process isn't orphaned when the launcher returns.
_game_process: Optional[subprocess.Popen] = None


def launch_cuphead(
    exe_path: Optional[str | Path] = None,
    wine_bin: str = "wine",
    wait_for_pid: bool = True,
    timeout: float = 30.0,
) -> Optional[int]:
    """Launch Cuphead via Wine as a child process.

    By launching the game ourselves, the Wine process tree becomes our
    child — and Linux allows process_vm_readv on child processes even
    when ptrace_scope=1 (the default security setting).

    Args:
        exe_path: Path to Cuphead.exe. Auto-detected if None.
        wine_bin: Wine binary name/path (default "wine").
        wait_for_pid: If True, wait until the actual game process appears.
        timeout: Max seconds to wait for the game process.

    Returns:
        PID of the Cuphead.exe Wine process, or None on failure.
    """
    global _game_process

    # Kill any previously launched game to prevent orphaned processes
    if _game_process is not None:
        logger.warning("Killing previously launched game (PID %d)", _game_process.pid)
        kill_cuphead()

    if exe_path is None:
        # Auto-detect from project structure
        project_root = Path(__file__).resolve().parent.parent
        exe_path = project_root / "game" / "CupHead" / "Cuphead.exe"

    exe_path = Path(exe_path)
    if not exe_path.exists():
        logger.error("Game executable not found: %s", exe_path)
        return None

    game_dir = exe_path.parent
    logger.info("Launching %s via %s ...", exe_path.name, wine_bin)

    # Launch Wine with the game exe, working dir set to game folder
    _game_process = subprocess.Popen(
        [wine_bin, str(exe_path)],
        cwd=str(game_dir),
        # Detach stdio so the game doesn't block on our terminal
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Keep in our process group so it's a child
        preexec_fn=None,
    )
    logger.info("Wine launcher PID: %d", _game_process.pid)

    if not wait_for_pid:
        return _game_process.pid

    # Wait for the actual Cuphead.exe process to appear
    # (Wine spawns several helper processes before the game itself)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        pid = find_wine_pid(exe_path.name)
        if pid is not None:
            logger.info(
                "Game process ready: PID %d (%.1fs)", pid, time.monotonic() - start
            )
            return pid
        time.sleep(0.5)

    logger.error("Timed out waiting for %s to start (%.0fs)", exe_path.name, timeout)
    return None


def kill_cuphead() -> None:
    """Terminate the launched Cuphead process tree."""
    global _game_process
    if _game_process is not None:
        try:
            _game_process.terminate()
            _game_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Game did not terminate gracefully, killing")
            try:
                _game_process.kill()
                _game_process.wait(timeout=3)
            except Exception as e:
                logger.debug("kill() failed: %s", e)
        except (ProcessLookupError, PermissionError) as e:
            logger.debug("terminate() failed (already dead?): %s", e)
        except Exception as e:
            logger.debug("Unexpected error terminating game: %s", e)
        _game_process = None
        logger.info("Cuphead terminated")


# ---------------------------------------------------------------------------
# PID finding
# ---------------------------------------------------------------------------


def find_wine_pid(exe_name: str = "Cuphead.exe") -> Optional[int]:
    """Find the PID of a Wine process by executable name.

    Scans /proc/*/cmdline for processes whose command line contains
    the given exe name. Returns the PID of the main Wine process
    (the one actually running the game, not wineserver).

    Args:
        exe_name: The Windows executable name to search for.

    Returns:
        PID as int, or None if not found.
    """
    proc = Path("/proc")
    candidates: list[tuple[int, str]] = []

    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue

        # cmdline is null-separated
        cmdline_str = cmdline.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        if exe_name.lower() in cmdline_str.lower():
            pid = int(entry.name)
            # Skip wineserver, wine-preloader etc. — we want the actual game
            if "wineserver" in cmdline_str.lower():
                continue
            candidates.append((pid, cmdline_str.strip()))

    if not candidates:
        return None

    # If multiple matches, pick the one with the highest PID (usually the
    # actual game process spawned after wine-preloader)
    candidates.sort(key=lambda x: x[0], reverse=True)
    pid, cmdline = candidates[0]
    logger.info("Found Wine process PID %d: %s", pid, cmdline[:120])
    return pid


# ---------------------------------------------------------------------------
# ProcessMemory — main memory reading class
# ---------------------------------------------------------------------------


class ProcessMemory:
    """Read memory from a remote process using process_vm_readv.

    Usage:
        pm = ProcessMemory(pid)
        value = pm.read_int32(0x7f1234567890)
        data = pm.read_bytes(address, 256)

    The class also provides maps parsing and module lookup for finding
    where specific DLLs/SOs are loaded in the process's address space.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._regions: list[MemoryRegion] = []
        self._regions_loaded = False

    # -- Raw read via process_vm_readv --

    # Maximum single read size to prevent accidental OOM from bad addresses/sizes
    MAX_READ_SIZE = 64 * 1024 * 1024  # 64 MB

    def read_bytes(self, address: int, size: int) -> Optional[bytes]:
        """Read `size` bytes from `address` in the target process.

        Returns None on failure (bad address, permission denied, size > 64MB, etc.).
        """
        if size <= 0 or address <= 0:
            return None
        if size > self.MAX_READ_SIZE:
            logger.error("read_bytes: size %d exceeds max %d", size, self.MAX_READ_SIZE)
            return None

        buf = (ctypes.c_char * size)()
        local = _iovec(ctypes.cast(buf, ctypes.c_void_p), size)
        remote = _iovec(ctypes.c_void_p(address), size)

        result = _process_vm_readv(
            self.pid,
            ctypes.byref(local),
            1,
            ctypes.byref(remote),
            1,
            0,
        )

        if result == -1:
            errno = ctypes.get_errno()
            if errno != 3:  # ESRCH = process gone, don't spam log
                logger.debug(
                    "process_vm_readv failed at 0x%x size=%d errno=%d",
                    address,
                    size,
                    errno,
                )
            return None

        if result != size:
            logger.debug(
                "process_vm_readv partial read at 0x%x: got %d of %d",
                address,
                result,
                size,
            )
            return bytes(buf[:result])

        return bytes(buf)

    # -- Typed read helpers --

    def read_int8(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 1)
        return struct.unpack("<b", data)[0] if data and len(data) == 1 else None

    def read_uint8(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 1)
        return struct.unpack("<B", data)[0] if data and len(data) == 1 else None

    def read_int16(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 2)
        return struct.unpack("<h", data)[0] if data and len(data) == 2 else None

    def read_uint16(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 2)
        return struct.unpack("<H", data)[0] if data and len(data) == 2 else None

    def read_int32(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 4)
        return struct.unpack("<i", data)[0] if data and len(data) == 4 else None

    def read_uint32(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 4)
        return struct.unpack("<I", data)[0] if data and len(data) == 4 else None

    def read_int64(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 8)
        return struct.unpack("<q", data)[0] if data and len(data) == 8 else None

    def read_uint64(self, address: int) -> Optional[int]:
        data = self.read_bytes(address, 8)
        return struct.unpack("<Q", data)[0] if data and len(data) == 8 else None

    def read_float(self, address: int) -> Optional[float]:
        data = self.read_bytes(address, 4)
        return struct.unpack("<f", data)[0] if data and len(data) == 4 else None

    def read_double(self, address: int) -> Optional[float]:
        data = self.read_bytes(address, 8)
        return struct.unpack("<d", data)[0] if data and len(data) == 8 else None

    def read_bool(self, address: int) -> Optional[bool]:
        data = self.read_bytes(address, 1)
        return data[0] != 0 if data and len(data) == 1 else None

    def read_pointer(self, address: int) -> Optional[int]:
        """Read a 64-bit pointer value."""
        return self.read_uint64(address)

    def read_string_utf16(self, address: int, max_len: int = 256) -> Optional[str]:
        """Read a .NET/Mono UTF-16LE string.

        Mono strings in memory have this layout (64-bit):
            +0x00: vtable pointer (8 bytes)
            +0x08: monitor/sync (8 bytes)
            +0x10: length as int32 (4 bytes)
            +0x14: padding (4 bytes, alignment)
            +0x18: char data starts (UTF-16LE)

        Wait — actually for Mono 64-bit the layout is:
            +0x00: vtable ptr (8)
            +0x08: sync block (8)
            +0x10: int32 length (4)
            +0x14: char[0] start (UTF-16LE, no padding)

        The `address` should point to the string object (not the char data).
        """
        # Read the length field
        length = self.read_int32(address + 0x10)
        if length is None or length < 0 or length > max_len:
            return None

        if length == 0:
            return ""

        # Read char data (UTF-16LE, 2 bytes per char)
        char_data = self.read_bytes(address + 0x14, length * 2)
        if char_data is None:
            return None

        try:
            return char_data.decode("utf-16-le")
        except UnicodeDecodeError:
            return None

    def read_pointer_chain(self, base: int, *offsets: int) -> Optional[int]:
        """Follow a chain of pointer dereferences.

        Starting at `base`, for each offset: read a pointer at (current + offset),
        making the read value the new current. All offsets are dereferenced,
        including the last — the return value is the pointer read at the final step.

        Example: read_pointer_chain(0x1000, 0x10, 0x20)
          1. ptr = read_pointer(0x1000 + 0x10) → e.g. 0x2000
          2. ptr = read_pointer(0x2000 + 0x20) → e.g. 0x3000
          3. return 0x3000

        Returns None if any dereference yields null or fails.
        """
        current = base
        for i, offset in enumerate(offsets):
            ptr = self.read_pointer(current + offset)
            if ptr is None or ptr == 0:
                logger.debug(
                    "Pointer chain broke at step %d: 0x%x + 0x%x → null",
                    i,
                    current,
                    offset,
                )
                return None
            current = ptr
        return current

    # -- /proc/<pid>/maps parsing --

    def _load_regions(self) -> None:
        """Parse /proc/<pid>/maps into MemoryRegion list."""
        self._regions = []
        maps_path = Path(f"/proc/{self.pid}/maps")
        try:
            text = maps_path.read_text()
        except (PermissionError, FileNotFoundError, ProcessLookupError) as e:
            logger.error("Cannot read %s: %s", maps_path, e)
            return

        for line in text.splitlines():
            m = _MAPS_RE.match(line)
            if not m:
                continue
            self._regions.append(
                MemoryRegion(
                    start=int(m.group(1), 16),
                    end=int(m.group(2), 16),
                    perms=m.group(3),
                    offset=int(m.group(4), 16),
                    device=m.group(5),
                    inode=int(m.group(6)),
                    pathname=m.group(7).strip(),
                )
            )

        self._regions_loaded = True
        logger.info("Loaded %d memory regions for PID %d", len(self._regions), self.pid)

    @property
    def regions(self) -> list[MemoryRegion]:
        """Lazy-loaded list of memory regions."""
        if not self._regions_loaded:
            self._load_regions()
        return self._regions

    def refresh_regions(self) -> None:
        """Force re-read of /proc/<pid>/maps."""
        self._regions_loaded = False
        self._regions = []

    def find_module(self, name: str) -> list[MemoryRegion]:
        """Find all memory regions belonging to a module (DLL/SO).

        Args:
            name: Substring to match in the pathname (case-insensitive).
                  E.g. "mono.dll", "libmono", "Cuphead.exe"

        Returns:
            List of MemoryRegion objects for matching regions.
        """
        name_lower = name.lower()
        return [r for r in self.regions if name_lower in r.pathname.lower()]

    def get_module_base(self, name: str) -> Optional[int]:
        """Get the base (lowest) address of a module.

        Returns None if the module isn't found.
        """
        regions = self.find_module(name)
        if not regions:
            return None
        return min(r.start for r in regions)

    def get_module_span(self, name: str) -> Optional[tuple[int, int]]:
        """Get (base_address, total_size) spanning all regions of a module.

        This gives the full address range from the lowest to highest mapping
        of the module, which is what you need for AOB scanning.
        """
        regions = self.find_module(name)
        if not regions:
            return None
        base = min(r.start for r in regions)
        end = max(r.end for r in regions)
        return (base, end - base)

    # -- Array-of-Bytes (AOB) scanning --

    def aob_scan(
        self,
        pattern: str | bytes,
        start: Optional[int] = None,
        size: Optional[int] = None,
        module: Optional[str] = None,
        readable_only: bool = True,
        first_only: bool = True,
    ) -> list[int]:
        """Scan for a byte pattern in the target process's memory.

        An AOB (Array of Bytes) pattern is a sequence of hex bytes where
        '??' or '?' means "match any byte". This is how game hackers find
        code/data structures that might be at different addresses each run,
        but always have the same instruction bytes nearby.

        Example patterns:
            "48 8B 05 ?? ?? ?? ?? 48 85 C0"  — x86-64 mov rax, [rip+??]
            "55 48 89 E5"                     — function prologue

        Args:
            pattern: Hex string with spaces (wildcards: ?? or ?), or raw bytes.
            start: Start address to scan from.
            size: Number of bytes to scan.
            module: If given, scan only this module's regions.
            readable_only: Only scan readable regions.
            first_only: Stop after first match.

        Returns:
            List of addresses where the pattern was found.
        """
        # Parse pattern into (bytes_to_match, mask) where mask[i] = True means
        # "this byte must match"
        if isinstance(pattern, str):
            match_bytes, mask = _parse_aob_pattern(pattern)
        else:
            match_bytes = pattern
            mask = [True] * len(pattern)

        pattern_len = len(match_bytes)
        if pattern_len == 0:
            return []

        # Determine scan regions
        if module:
            scan_regions = self.find_module(module)
            if not scan_regions:
                logger.warning("Module '%s' not found for AOB scan", module)
                return []
        elif start is not None and size is not None:
            # Scan a specific range — find overlapping regions
            scan_end = start + size
            scan_regions = [
                r for r in self.regions if r.start < scan_end and r.end > start
            ]
        else:
            scan_regions = list(self.regions)

        if readable_only:
            scan_regions = [r for r in scan_regions if r.readable]

        results: list[int] = []
        chunk_size = 4 * 1024 * 1024  # Read 4 MB at a time

        for region in scan_regions:
            r_start = region.start
            if start is not None:
                r_start = max(r_start, start)
            r_end = region.end
            if start is not None and size is not None:
                r_end = min(r_end, start + size)

            offset = 0
            total = r_end - r_start
            overlap = pattern_len - 1  # Overlap to catch patterns spanning chunks

            while offset < total:
                read_size = min(chunk_size + overlap, total - offset)
                data = self.read_bytes(r_start + offset, read_size)
                if data is None:
                    offset += chunk_size
                    continue

                # Scan this chunk
                for i in range(len(data) - pattern_len + 1):
                    found = True
                    for j in range(pattern_len):
                        if mask[j] and data[i + j] != match_bytes[j]:
                            found = False
                            break
                    if found:
                        addr = r_start + offset + i
                        results.append(addr)
                        if first_only:
                            return results

                offset += chunk_size

        return results

    def aob_scan_module(
        self,
        module: str,
        pattern: str,
        first_only: bool = True,
    ) -> list[int]:
        """Convenience: AOB scan within a specific module."""
        return self.aob_scan(pattern, module=module, first_only=first_only)

    # -- Process status --

    def is_alive(self) -> bool:
        """Check if the target process is still running."""
        return Path(f"/proc/{self.pid}").exists()

    def __repr__(self) -> str:
        return f"ProcessMemory(pid={self.pid})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_aob_pattern(pattern: str) -> tuple[bytes, list[bool]]:
    """Parse an AOB pattern string into (bytes, mask).

    Pattern format: space-separated hex bytes, '??' or '?' for wildcards.
    Example: "48 8B 05 ?? ?? ?? ?? 48 85 C0"

    Returns:
        (match_bytes, mask) where mask[i] is True if byte must match.
    """
    tokens = pattern.strip().split()
    match_bytes = bytearray()
    mask: list[bool] = []

    for token in tokens:
        if token in ("??", "?", "**"):
            match_bytes.append(0)
            mask.append(False)
        else:
            match_bytes.append(int(token, 16))
            mask.append(True)

    return bytes(match_bytes), mask


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Support --launch flag to start the game
    do_launch = "--launch" in sys.argv

    exe = "Cuphead.exe"
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            exe = arg
            break

    pid = find_wine_pid(exe)

    if pid is None and do_launch:
        print("Game not running. Launching via Wine...")
        pid = launch_cuphead()

    if pid is None:
        print(f"Could not find Wine process for '{exe}'")
        print("Tip: use --launch to start the game, or launch it manually")
        sys.exit(1)

    pm = ProcessMemory(pid)
    print(f"\n{pm}")
    print(f"Process alive: {pm.is_alive()}")
    print(f"Total regions: {len(pm.regions)}")

    # Show module summary
    modules: dict[str, list[MemoryRegion]] = {}
    for r in pm.regions:
        if r.pathname:
            key = Path(r.pathname).name
            modules.setdefault(key, []).append(r)

    print(f"\nMapped modules ({len(modules)}):")
    for name, regs in sorted(modules.items()):
        total = sum(r.size for r in regs)
        base = min(r.start for r in regs)
        print(
            f"  {name:40s} base=0x{base:012x}  size={total:>10,} bytes  ({len(regs)} regions)"
        )

    # Look for mono
    mono = pm.find_module("mono")
    if mono:
        base = min(r.start for r in mono)
        total = sum(r.size for r in mono)
        print(f"\nMono found: base=0x{base:012x}, total mapped={total:,} bytes")
    else:
        print("\nMono NOT found in mapped modules")
