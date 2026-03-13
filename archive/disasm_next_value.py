"""
Disassemble the next_value function used by MonoInternalHashTable for the class cache.

Known offsets (from previous analysis):
  - MonoInternalHashTable (class cache): MonoImage + 0x3D0
  - hash_func:    MonoImage + 0x3D0 + 0x00 = MonoImage + 0x3D0
  - key_extract:  MonoImage + 0x3D0 + 0x08 = MonoImage + 0x3D8
  - next_value:   MonoImage + 0x3D0 + 0x10 = MonoImage + 0x3E0
  - table_size:   MonoImage + 0x3D0 + 0x18 = MonoImage + 0x3E8
  - entry_count:  MonoImage + 0x3D0 + 0x1C = MonoImage + 0x3EC
  - table:        MonoImage + 0x3D0 + 0x20 = MonoImage + 0x3F0

Goal: Find the MonoClass offset for the hash chain next pointer.
"""

import pefile
import struct
from capstone import Cs, CS_ARCH_X86, CS_MODE_64

MONO_DLL = "/home/biel/workspace/paper_engine/game/CupHead/Cuphead_Data/Mono/EmbedRuntime/mono.dll"


def disasm_range(pe, md, rva, size, label=None):
    """Disassemble a range and return list of (address, mnemonic, op_str) plus print."""
    data = pe.get_data(rva, size)
    image_base = pe.OPTIONAL_HEADER.ImageBase
    instructions = []
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    for i in md.disasm(data, image_base + rva):
        instructions.append((i.address, i.mnemonic, i.op_str, i.size))
        line = f"  0x{i.address:X}  {i.mnemonic:8s} {i.op_str}"
        print(line)
        if i.mnemonic == "int3" and instructions and len(instructions) > 2:
            break
    return instructions


def resolve_rip_relative(instr_va, instr_size, disp):
    """Resolve a RIP-relative address."""
    return instr_va + instr_size + disp


def main():
    pe = pefile.PE(MONO_DLL)
    image_base = pe.OPTIONAL_HEADER.ImageBase
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True

    print("=" * 60)
    print("  MonoClass Hash Chain Next-Pointer Offset Finder")
    print("  mono.dll from Cuphead (Unity 2017.4 / Mono ~5.11)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Locate mono_image_init (exported)
    # ---------------------------------------------------------------
    image_init_rva = None
    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
        name = exp.name.decode() if exp.name else ""
        if name == "mono_image_init":
            image_init_rva = exp.address
            break

    if not image_init_rva:
        print("ERROR: mono_image_init not found in exports!")
        return

    print(
        f"\nmono_image_init: RVA 0x{image_init_rva:X} (VA 0x{image_base + image_init_rva:X})"
    )

    # ---------------------------------------------------------------
    # Step 2: Disassemble mono_image_init to find hash table setup
    # ---------------------------------------------------------------
    init_instrs = disasm_range(
        pe, md, image_init_rva, 0xC0, "mono_image_init — hash table initialization"
    )

    # Find the call to mono_internal_hash_table_init
    # Look for: lea rcx, [rbx + 0x3D0]  (hash table address)
    # Followed by: lea r9, [rip + XX]    (next_value)
    #              lea r8, [rip + XX]    (key_extract)
    #              lea rdx, [rip + XX]   (hash_func)
    #              call XXX              (mono_internal_hash_table_init)

    hash_func_va = None
    key_extract_va = None
    next_value_va = None
    init_call_va = None

    for idx, (addr, mnem, ops, size) in enumerate(init_instrs):
        # lea rcx, [rbx + 0x3d0] -> hash table pointer
        if mnem == "lea" and "0x3d0" in ops.lower() and ops.startswith("rcx"):
            print(f"\n  >> Hash table base: {ops}")

        # Resolve LEA r9 (next_value)
        if mnem == "lea" and ops.startswith("r9"):
            # Parse RIP-relative displacement
            raw_data = pe.get_data(addr - image_base, size)
            if size == 7:  # LEA r9, [rip + disp32]
                disp = struct.unpack_from("<i", raw_data, 3)[0]
                next_value_va = resolve_rip_relative(addr, size, disp)
                print(
                    f"\n  >> next_value function: VA 0x{next_value_va:X} "
                    f"(RVA 0x{next_value_va - image_base:X})"
                )

        # Resolve LEA r8 (key_extract)
        if mnem == "lea" and ops.startswith("r8"):
            raw_data = pe.get_data(addr - image_base, size)
            if size == 7:
                disp = struct.unpack_from("<i", raw_data, 3)[0]
                key_extract_va = resolve_rip_relative(addr, size, disp)
                print(
                    f"  >> key_extract function: VA 0x{key_extract_va:X} "
                    f"(RVA 0x{key_extract_va - image_base:X})"
                )

        # Resolve LEA rdx (hash_func)
        if mnem == "lea" and ops.startswith("rdx"):
            raw_data = pe.get_data(addr - image_base, size)
            if size == 7:
                disp = struct.unpack_from("<i", raw_data, 3)[0]
                hash_func_va = resolve_rip_relative(addr, size, disp)
                print(
                    f"  >> hash_func function:   VA 0x{hash_func_va:X} "
                    f"(RVA 0x{hash_func_va - image_base:X})"
                )

    # ---------------------------------------------------------------
    # Step 3: Disassemble next_value function
    # ---------------------------------------------------------------
    if next_value_va:
        next_value_rva = next_value_va - image_base
        instrs = disasm_range(
            pe,
            md,
            next_value_rva,
            16,
            f"next_value (class_next_value) @ RVA 0x{next_value_rva:X}",
        )

        # Extract the offset from the LEA instruction
        chain_offset = None
        for addr, mnem, ops, size in instrs:
            if mnem == "lea" and ops.startswith("rax"):
                # Parse: lea rax, [rcx + OFFSET]
                if "+" in ops:
                    offset_str = ops.split("+")[-1].strip().rstrip("]").strip()
                    chain_offset = int(offset_str, 16)
                    break

        if chain_offset is not None:
            print(f"\n  >>> next_value returns &(MonoClass + 0x{chain_offset:X})")
            print(f"  >>> Hash chain next pointer is at MonoClass + 0x{chain_offset:X}")

    # ---------------------------------------------------------------
    # Step 4: Disassemble key_extract function
    # ---------------------------------------------------------------
    if key_extract_va:
        key_extract_rva = key_extract_va - image_base
        instrs = disasm_range(
            pe,
            md,
            key_extract_rva,
            16,
            f"key_extract (class_key_extract) @ RVA 0x{key_extract_rva:X}",
        )

        for addr, mnem, ops, size in instrs:
            if mnem == "mov" and "rcx" in ops:
                if "+" in ops:
                    offset_str = ops.split("+")[-1].strip().rstrip("]").strip()
                    key_offset = int(offset_str, 16)
                    print(
                        f"\n  >>> key_extract returns *(MonoClass + 0x{key_offset:X})"
                    )
                    print(f"  >>> Confirms type_token at MonoClass + 0x{key_offset:X}")

    # ---------------------------------------------------------------
    # Step 5: Disassemble hash_func
    # ---------------------------------------------------------------
    if hash_func_va:
        hash_func_rva = hash_func_va - image_base
        disasm_range(
            pe,
            md,
            hash_func_rva,
            16,
            f"hash_func (identity hash) @ RVA 0x{hash_func_rva:X}",
        )

    # ---------------------------------------------------------------
    # Step 6: Disassemble mono_internal_hash_table_init
    # ---------------------------------------------------------------
    init_func_rva = 0xD66C
    disasm_range(
        pe,
        md,
        init_func_rva,
        0x40,
        f"mono_internal_hash_table_init @ RVA 0x{init_func_rva:X}",
    )

    # ---------------------------------------------------------------
    # Step 7: Disassemble sub_d6c0 (hash table lookup with walk loop)
    # ---------------------------------------------------------------
    lookup_rva = 0xD6C0
    print()
    instrs = disasm_range(
        pe,
        md,
        lookup_rva,
        0x90,
        f"sub_d6c0 — mono_internal_hash_table_lookup @ RVA 0x{lookup_rva:X}",
    )

    # Annotate the hash walk loop
    print(f"\n{'=' * 60}")
    print("  Annotated Hash Walk Loop")
    print(f"{'=' * 60}")
    annotations = {
        0x18000D6CF: "if (table->table == NULL) -> error",
        0x18000D6D4: "rsi = key (type_token)",
        0x18000D6D7: "rdi = hash_table (MonoInternalHashTable*)",
        0x18000D70A: "rcx = key (type_token)",
        0x18000D70D: "eax = hash_func(key)  → call [rdi+0x00]",
        0x18000D70F: "zero rdx for div",
        0x18000D711: "edx = hash % table_size  → div [rdi+0x18]",
        0x18000D714: "rax = table->table",
        0x18000D718: "rbx = table[hash % size] → first node in bucket",
        0x18000D71C: "jump to null check",
        0x18000D71E: "--- loop body: rcx = current node (MonoClass*) ---",
        0x18000D721: "rax = key_extract(node) → call [rdi+0x08] → MonoClass+0x60",
        0x18000D724: "compare extracted key with search key",
        0x18000D727: "if match → found!",
        0x18000D729: "rcx = current node",
        0x18000D72C: "rax = next_value(node) → call [rdi+0x10] → &(MonoClass+0x108)",
        0x18000D72F: "rbx = *rax = *(MonoClass+0x108) → next node in chain",
        0x18000D732: "--- loop test: is current node NULL? ---",
        0x18000D735: "if not null → continue loop",
        0x18000D737: "return NULL (not found)",
        0x18000D749: "return rbx (found match)",
    }

    for addr, mnem, ops, size in instrs:
        if addr in annotations:
            note = annotations[addr]
            print(f"  0x{addr:X}  {mnem:8s} {ops:40s} ; {note}")
        if addr > 0x18000D74C:
            break

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print()
    print("  MonoInternalHashTable layout (class cache):")
    print("    +0x00  hash_func      → identity (mov eax, ecx; ret)")
    print("    +0x08  key_extract    → reads MonoClass+0x60 (type_token)")
    print("    +0x10  next_value     → returns &(MonoClass+0x108)")
    print("    +0x18  table_size     (uint32)")
    print("    +0x1C  entry_count    (uint32)")
    print("    +0x20  table          (void** bucket array)")
    print()
    if chain_offset is not None:
        print(f"  Hash chain next pointer:  MonoClass + 0x{chain_offset:X}")
        print(f"  Type token (key):         MonoClass + 0x60")
        print()
        print(f"  The next_value function returns a pointer to the field,")
        print(
            f"  and the lookup loop dereferences it: *(MonoClass + 0x{chain_offset:X})"
        )
        print(f"  to walk to the next MonoClass* in the hash bucket chain.")
    print()


if __name__ == "__main__":
    main()
