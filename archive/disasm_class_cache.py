#!/usr/bin/env python3
"""
Disassemble mono_class_get and related internal functions from mono.dll
to find where MonoClass* pointers are cached.

Target: Cuphead's mono.dll (PE64, x86_64, Wine/Windows)
"""

import pefile
import struct
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_CALL, CS_GRP_RET

DLL_PATH = "/home/biel/workspace/paper_engine/game/CupHead/Cuphead_Data/Mono/EmbedRuntime/mono.dll"

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def load_pe():
    pe = pefile.PE(DLL_PATH)
    return pe


def build_export_map(pe):
    """name -> RVA"""
    m = {}
    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
        if exp.name:
            m[exp.name.decode()] = exp.address
    return m


def rva_to_offset(pe, rva):
    """Convert RVA to file offset."""
    for section in pe.sections:
        if (
            section.VirtualAddress
            <= rva
            < section.VirtualAddress + section.Misc_VirtualSize
        ):
            return rva - section.VirtualAddress + section.PointerToRawData
    return None


def read_bytes_at_rva(pe, rva, size):
    """Read `size` bytes from the file at the given RVA."""
    off = rva_to_offset(pe, rva)
    if off is None:
        return None
    return pe.__data__[off : off + size]


def va(pe, rva):
    return pe.OPTIONAL_HEADER.ImageBase + rva


# ──────────────────────────────────────────────────────────────────────
# Disassembler
# ──────────────────────────────────────────────────────────────────────


def disassemble_function(pe, rva, max_bytes=2048, label=""):
    """
    Disassemble starting at `rva` until we hit a RET or exceed max_bytes.
    Returns list of (address, mnemonic, op_str, bytes_hex, size) and the raw
    capstone instruction objects.
    """
    code = read_bytes_at_rva(pe, rva, max_bytes)
    if code is None:
        print(f"  [!] Cannot read bytes at RVA 0x{rva:x}")
        return [], []

    base_va = va(pe, rva)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    results = []
    insns = []
    ret_count = 0
    for insn in md.disasm(code, base_va):
        bhex = insn.bytes.hex()
        results.append((insn.address, insn.mnemonic, insn.op_str, bhex, insn.size))
        insns.append(insn)
        # Stop after first unconditional ret that isn't followed by more code
        if CS_GRP_RET in insn.groups:
            ret_count += 1
            # Heuristic: stop after 2nd ret or if we see int3 / padding after ret
            if ret_count >= 3:
                break
        # Also stop on int3 (padding)
        if insn.mnemonic == "int3":
            break

    return results, insns


def print_disasm(results, label=""):
    if label:
        print(f"\n{'=' * 80}")
        print(f"  {label}")
        print(f"{'=' * 80}")
    for addr, mn, ops, bhex, sz in results:
        print(f"  0x{addr:016x}:  {bhex:<24s}  {mn:<10s} {ops}")


# ──────────────────────────────────────────────────────────────────────
# Analysis: find CALL targets and memory store patterns
# ──────────────────────────────────────────────────────────────────────


def find_call_targets(pe, insns):
    """Return set of RVAs that are CALL targets (direct calls only)."""
    targets = set()
    for insn in insns:
        if CS_GRP_CALL in insn.groups:
            # Direct call: the operand is an immediate address
            if insn.operands and insn.operands[0].type == 2:  # IMM
                target_va = insn.operands[0].imm
                target_rva = target_va - pe.OPTIONAL_HEADER.ImageBase
                if 0 < target_rva < pe.OPTIONAL_HEADER.SizeOfImage:
                    targets.add(target_rva)
    return targets


def resolve_rva_to_export(pe, export_map, target_rva):
    """Check if an RVA matches a known export."""
    inv = {v: k for k, v in export_map.items()}
    return inv.get(target_rva, None)


def find_store_instructions(insns):
    """Find MOV [mem], reg instructions that could be cache stores."""
    stores = []
    for insn in insns:
        mn = insn.mnemonic
        if mn in ("mov", "movq", "xchg", "cmpxchg"):
            # Check if first operand is memory
            if insn.operands and insn.operands[0].type == 3:  # MEM
                stores.append(insn)
    return stores


# ──────────────────────────────────────────────────────────────────────
# Deep disassembly: follow call chains
# ──────────────────────────────────────────────────────────────────────


def disassemble_deep(
    pe,
    export_map,
    start_rva,
    label,
    depth=0,
    visited=None,
    max_depth=3,
    max_fn_bytes=4096,
):
    """
    Disassemble a function and recursively follow internal (non-exported) CALL targets.
    """
    if visited is None:
        visited = set()
    if start_rva in visited or depth > max_depth:
        return
    visited.add(start_rva)

    results, insns = disassemble_function(
        pe, start_rva, max_bytes=max_fn_bytes, label=label
    )
    indent = "  " * depth
    print(f"\n{'=' * 80}")
    print(
        f"{indent}FUNCTION: {label}  (RVA=0x{start_rva:x}  VA=0x{va(pe, start_rva):x})"
    )
    print(f"{'=' * 80}")
    for addr, mn, ops, bhex, sz in results:
        print(f"{indent}  0x{addr:016x}:  {bhex:<24s}  {mn:<10s} {ops}")

    # Find store instructions
    stores = find_store_instructions(insns)
    if stores:
        print(f"\n{indent}  --- Memory store instructions ---")
        for s in stores:
            mem = s.operands[0].mem
            base_reg = s.reg_name(mem.base) if mem.base else "none"
            index_reg = s.reg_name(mem.index) if mem.index else "none"
            disp = mem.disp
            scale = mem.scale
            src = s.op_str.split(",")[-1].strip() if "," in s.op_str else "?"
            detail = f"base={base_reg} index={index_reg} scale={scale} disp=0x{disp:x}"
            print(
                f"{indent}  0x{s.address:x}: {s.mnemonic} {s.op_str}  ;; {detail} (src={src})"
            )

    # Follow call targets
    call_targets = find_call_targets(pe, insns)
    for target_rva in sorted(call_targets):
        exp_name = resolve_rva_to_export(pe, export_map, target_rva)
        if exp_name:
            # Don't recurse into well-known exported functions, just note them
            print(f"\n{indent}  -> CALLS exported: {exp_name} (RVA=0x{target_rva:x})")
        else:
            # Internal function — recurse
            sub_label = f"sub_{target_rva:x}"
            disassemble_deep(
                pe,
                export_map,
                target_rva,
                sub_label,
                depth=depth + 1,
                visited=visited,
                max_depth=max_depth,
                max_fn_bytes=max_fn_bytes,
            )


# ──────────────────────────────────────────────────────────────────────
# Targeted: find the internal typedef->class function
# ──────────────────────────────────────────────────────────────────────


def scan_for_class_create_from_typedef(pe, export_map):
    """
    mono_class_get typically does:
      1. Check cache (image->class_cache indexed by typedef token row)
      2. If miss, call mono_class_create_from_typedef / mono_class_from_typedef_checked
      3. Store result in cache
      4. Return

    We disassemble mono_class_get deeply to find this pattern.
    """
    rva = export_map.get("mono_class_get")
    if rva is None:
        print("[!] mono_class_get not found in exports!")
        return

    print("\n" + "#" * 80)
    print("# DEEP DISASSEMBLY: mono_class_get + internal callees")
    print("#" * 80)

    disassemble_deep(
        pe, export_map, rva, "mono_class_get", depth=0, max_depth=3, max_fn_bytes=4096
    )


# ──────────────────────────────────────────────────────────────────────
# Also disassemble mono_class_get_full (handles generic instantiation)
# ──────────────────────────────────────────────────────────────────────


def scan_class_get_full(pe, export_map):
    rva = export_map.get("mono_class_get_full")
    if rva is None:
        print("[!] mono_class_get_full not found in exports!")
        return

    print("\n" + "#" * 80)
    print("# DEEP DISASSEMBLY: mono_class_get_full + internal callees")
    print("#" * 80)

    disassemble_deep(
        pe,
        export_map,
        rva,
        "mono_class_get_full",
        depth=0,
        max_depth=2,
        max_fn_bytes=4096,
    )


# ──────────────────────────────────────────────────────────────────────
# String search: find "mono_class_create_from_typedef" in .rdata
# ──────────────────────────────────────────────────────────────────────


def find_string_refs(pe, needle):
    """Find file offsets where `needle` appears (for cross-reference)."""
    data = pe.__data__
    needle_bytes = needle.encode("ascii")
    results = []
    start = 0
    while True:
        idx = data.find(needle_bytes, start)
        if idx == -1:
            break
        # Convert file offset to RVA
        for section in pe.sections:
            if (
                section.PointerToRawData
                <= idx
                < section.PointerToRawData + section.SizeOfRawData
            ):
                rva = idx - section.PointerToRawData + section.VirtualAddress
                results.append((idx, rva))
                break
        start = idx + 1
    return results


def find_xrefs_to_rva(pe, target_rva, section_name=".text"):
    """
    Scan .text section for LEA instructions that reference a target RVA.
    We look for LEA reg, [rip+disp] patterns: 48 8d 0d XX XX XX XX (and variants).
    """
    text_section = None
    for s in pe.sections:
        if s.Name.rstrip(b"\x00").decode() == section_name:
            text_section = s
            break
    if text_section is None:
        return []

    data = pe.__data__
    start_off = text_section.PointerToRawData
    end_off = start_off + text_section.SizeOfRawData
    base_rva = text_section.VirtualAddress

    results = []
    # LEA with RIP-relative: opcode 8d with ModRM indicating RIP-relative (mod=00, rm=101)
    # REX prefix can be 48, 4c, etc.
    pos = start_off
    while pos < end_off - 7:
        b0 = data[pos]
        # Check for REX.W prefix (0x48-0x4f) followed by 0x8d
        if (b0 & 0xF0) == 0x40:
            b1 = data[pos + 1]
            if b1 == 0x8D:
                modrm = data[pos + 2]
                mod = (modrm >> 6) & 3
                rm = modrm & 7
                if mod == 0 and rm == 5:
                    # RIP-relative LEA
                    disp = struct.unpack_from("<i", data, pos + 3)[0]
                    insn_rva = (pos - start_off) + base_rva
                    # RIP-relative is relative to end of instruction (7 bytes)
                    ref_rva = insn_rva + 7 + disp
                    if ref_rva == target_rva:
                        results.append(insn_rva)
        pos += 1
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading mono.dll...")
    pe = load_pe()
    export_map = build_export_map(pe)

    print(f"Image base: 0x{pe.OPTIONAL_HEADER.ImageBase:x}")
    print(f"Total exports: {len(export_map)}")

    # List relevant exports
    print("\n--- Relevant exports ---")
    for name in sorted(export_map):
        if any(
            kw in name.lower()
            for kw in ("class_get", "class_from", "class_create", "typedef")
        ):
            print(f"  0x{export_map[name]:08x}  {name}")

    # 1) Deep disassembly of mono_class_get
    scan_for_class_create_from_typedef(pe, export_map)

    # 2) Also do mono_class_get_full
    scan_class_get_full(pe, export_map)

    # 3) Search for string "mono_class_create_from_typedef" to locate the internal function
    print("\n" + "#" * 80)
    print("# STRING SEARCH: locating internal function names")
    print("#" * 80)
    for needle in [
        "mono_class_create_from_typedef",
        "mono_class_from_typedef_checked",
        "class_cache",
        "type_cache",
    ]:
        refs = find_string_refs(pe, needle)
        if refs:
            for foff, rva in refs:
                print(
                    f"  String '{needle}' at file offset 0x{foff:x}, RVA 0x{rva:x}, VA 0x{va(pe, rva):x}"
                )
                # Find code that references this string
                xrefs = find_xrefs_to_rva(pe, rva)
                for xref_rva in xrefs[:5]:
                    print(
                        f"    Referenced by code at RVA 0x{xref_rva:x} VA 0x{va(pe, xref_rva):x}"
                    )
                    # Disassemble around the xref to find the function start
                    # Walk backwards to find function prologue
                    scan_start = max(0, xref_rva - 256)
                    code_bytes = read_bytes_at_rva(
                        pe, scan_start, xref_rva - scan_start + 128
                    )
                    if code_bytes:
                        # Look for common prologue patterns (push rbp / sub rsp)
                        md = Cs(CS_ARCH_X86, CS_MODE_64)
                        md.detail = True
                        # Simple approach: disassemble from xref-256, find the function boundary
                        func_start_rva = None
                        # Try to find int3 or ret+nop padding before the function
                        raw_before = read_bytes_at_rva(pe, max(0, xref_rva - 512), 512)
                        if raw_before:
                            # Scan backwards for int3 (0xCC) or nop (0x90) padding
                            for i in range(len(raw_before) - 1, -1, -1):
                                if raw_before[i] in (0xCC, 0xC3):
                                    candidate_rva = xref_rva - 512 + i + 1
                                    if candidate_rva < xref_rva and candidate_rva > 0:
                                        func_start_rva = candidate_rva
                                        break
                        if func_start_rva:
                            print(
                                f"    Likely function start at RVA 0x{func_start_rva:x}"
                            )
        else:
            print(f"  String '{needle}' NOT found")

    # 4) Specifically disassemble the function that mono_class_get calls internally
    print("\n" + "#" * 80)
    print("# ANALYSIS: mono_class_get call chain")
    print("#" * 80)

    rva = export_map.get("mono_class_get")
    results, insns = disassemble_function(pe, rva, max_bytes=512)

    # mono_class_get is typically a thin wrapper around mono_class_get_checked
    # or directly calls mono_class_from_typedef_checked
    call_targets = find_call_targets(pe, insns)
    print(f"\nmono_class_get calls these internal functions:")
    for t in sorted(call_targets):
        exp = resolve_rva_to_export(pe, export_map, t)
        name = exp if exp else f"sub_{t:x}"
        print(f"  RVA 0x{t:x}  VA 0x{va(pe, t):x}  {name}")

        if not exp:
            # Disassemble each internal callee with more bytes
            print(f"\n  --- Disassembly of {name} ---")
            sub_results, sub_insns = disassemble_function(pe, t, max_bytes=4096)
            for addr, mn, ops, bhex, sz in sub_results:
                print(f"    0x{addr:016x}:  {bhex:<24s}  {mn:<10s} {ops}")

            # Find stores
            stores = find_store_instructions(sub_insns)
            if stores:
                print(f"\n  --- Stores in {name} ---")
                for s in stores:
                    mem = s.operands[0].mem
                    base_reg = s.reg_name(mem.base) if mem.base else "none"
                    index_reg = s.reg_name(mem.index) if mem.index else "none"
                    disp = mem.disp
                    print(f"    0x{s.address:x}: {s.mnemonic} {s.op_str}")
                    print(
                        f"      base={base_reg} index={index_reg} scale={mem.scale} disp=0x{disp:x}"
                    )

            # Recurse one more level
            sub_calls = find_call_targets(pe, sub_insns)
            for st in sorted(sub_calls):
                exp2 = resolve_rva_to_export(pe, export_map, st)
                if not exp2:
                    print(f"\n  --- Sub-callee sub_{st:x} ---")
                    ss_results, ss_insns = disassemble_function(pe, st, max_bytes=4096)
                    for addr, mn, ops, bhex, sz in ss_results:
                        print(f"      0x{addr:016x}:  {bhex:<24s}  {mn:<10s} {ops}")
                    ss_stores = find_store_instructions(ss_insns)
                    if ss_stores:
                        print(f"\n    --- Stores in sub_{st:x} ---")
                        for s in ss_stores:
                            mem = s.operands[0].mem
                            base_reg = s.reg_name(mem.base) if mem.base else "none"
                            index_reg = s.reg_name(mem.index) if mem.index else "none"
                            disp = mem.disp
                            print(f"      0x{s.address:x}: {s.mnemonic} {s.op_str}")
                            print(
                                f"        base={base_reg} index={index_reg} scale={mem.scale} disp=0x{disp:x}"
                            )

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
