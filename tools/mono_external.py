"""
External Mono runtime reader — walks Mono's internal data structures via
process_vm_readv to find classes, fields, and game state without calling
any Mono API functions.

This module implements a pure-memory-reading approach to Mono introspection.
All struct offsets were reverse-engineered by disassembling mono.dll exports
with Capstone, matching them to Mono's public API function implementations.

Architecture:
    Python (parent) --process_vm_readv--> Wine child process memory
        mono.dll .data section --> root_domain global
        root_domain --> domain_assemblies (GSList)
        assembly --> image --> class_cache / TypeDef table
        class --> runtime_info --> vtable --> static_field_data
        class --> fields[] --> field offsets for instance data

Requires:
    - Game launched as a child process (for ptrace_scope=1 access)
    - mono.dll base address (from bridge shm or /proc/pid/maps)

Usage:
    from tools.memory_reader import ProcessMemory, find_wine_pid
    from tools.mono_external import MonoExternal

    pm = ProcessMemory(pid)
    mono = MonoExternal(pm, mono_base=0x6ffffc3e0000)
    mono.attach()

    # Find a class
    klass = mono.find_class("Assembly-CSharp", "", "PlayerStatsManager")
    print(klass)

    # Read static field
    level_class = mono.find_class("Assembly-CSharp", "", "Level")
    current = mono.read_static_field_ptr(level_class, "<Current>k__BackingField")
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mono struct offsets — reverse-engineered from mono.dll PE disassembly
# These are specific to the Mono version shipped with Unity 2017.4 (64-bit)
# ---------------------------------------------------------------------------


# Global variable RVAs (offsets from mono.dll base address)
class MonoGlobals:
    ROOT_DOMAIN_RVA = 0x2675E0  # MonoDomain* root_domain
    LOADED_ASSEMBLIES_RVA = 0x267310  # GSList* loaded_assemblies


# MonoDomain struct offsets
class MonoDomainOffsets:
    DOMAIN_ID = 0xBC  # int32 domain_id
    DOMAIN_ASSEMBLIES = 0xC8  # GSList* domain_assemblies


# GSList (GLib singly-linked list)
class GSListOffsets:
    DATA = 0x00  # void* data
    NEXT = 0x08  # GSList* next


# MonoAssembly struct offsets
class MonoAssemblyOffsets:
    ANAME_NAME = 0x10  # char* (within MonoAssemblyName)
    IMAGE = 0x58  # MonoImage*


# MonoImage struct offsets
class MonoImageOffsets:
    FLAGS = 0x1C  # uint8 (bit 2 = dynamic)
    NAME_FILENAME = 0x20  # char* (full filename)
    NAME = 0x28  # char* (short name)
    TABLE_BASE = 0xC0  # MonoTableInfo tables[] start
    TABLE_INFO_SIZE = 0x10  # sizeof(MonoTableInfo)
    TABLE_ID_OFFSET = 0x0C  # tables start at (table_id + 0x0C) * 0x10 from image
    CLASS_TABLE = 0x390  # MonoClass** (typeref class table, indexed by row)
    MODULES = 0x398  # MonoImage*[] modules
    NUM_MODULES = 0x3A0  # int num_modules
    REFERENCES = 0x3C0  # MonoAssembly** references
    # MonoInternalHashTable for typedef→MonoClass* cache (primary class cache)
    # Struct: +0x00 hash_func, +0x08 key_extract, +0x10 next_value,
    #         +0x18 num_buckets (uint32), +0x1C num_entries (uint32),
    #         +0x20 table (MonoClass** bucket array)
    CLASS_CACHE_HT = 0x3D0  # MonoInternalHashTable (typedef token → MonoClass*)
    CLASS_CACHE_HT_NUM_BUCKETS = 0x3E8  # uint32 (image + 0x3D0 + 0x18)
    CLASS_CACHE_HT_NUM_ENTRIES = 0x3EC  # uint32 (image + 0x3D0 + 0x1C)
    CLASS_CACHE_HT_TABLE = 0x3F0  # MonoClass** bucket array (image + 0x3D0 + 0x20)
    CLASS_CACHE = 0x428  # GHashTable* name_cache (namespace → name → token)


# MonoTableInfo struct offsets
class MonoTableInfoOffsets:
    BASE = 0x00  # const char* base (pointer to raw table data)
    ROWS_AND_FLAGS = 0x08  # uint32 (rows = value & 0xFFFFFF)
    ROW_SIZE = 0x0C  # uint32 row_size


# MonoClass struct offsets
class MonoClassOffsets:
    ELEMENT_CLASS = 0x00  # MonoClass* (for arrays)
    BITFIELDS = 0x20  # uint32 (is_valuetype at bit 3)
    INIT_BYTE = 0x2C  # uint8 (class init state)
    PARENT = 0x30  # MonoClass* parent
    IMAGE = 0x48  # MonoImage* image
    NAME = 0x50  # char* name
    NAMESPACE = 0x58  # char* name_space
    TYPE_TOKEN = 0x60  # uint32 type_token (TypeDef table token)
    FIELD_COUNT = 0x9C  # uint32 field.count
    FIELDS = 0xB0  # MonoClassField* fields (array)
    THIS_ARG = 0xD0  # MonoType this_arg (inline struct)
    RUNTIME_INFO = 0x100  # MonoClassRuntimeInfo*
    NEXT_CLASS_CACHE = 0x108  # MonoClass* (chain link in MonoInternalHashTable)


# MonoClassField struct (0x20 bytes each)
class MonoClassFieldOffsets:
    SIZEOF = 0x20  # sizeof(MonoClassField)
    TYPE = 0x00  # MonoType* type
    NAME = 0x08  # char* name
    PARENT = 0x10  # MonoClass* parent (the defining class)
    OFFSET = 0x18  # uint32 offset (byte offset from object start)


# MonoType struct offsets
class MonoTypeOffsets:
    DATA = 0x00  # union { MonoClass* klass; MonoType* type; ... }
    ATTRS = 0x08  # uint16 attrs (field attributes, including static flag)
    TYPE = 0x0A  # int8 type (MONO_TYPE_* enum)
    # Field attribute flags (in ATTRS)
    FIELD_ATTR_STATIC = 0x0010
    FIELD_ATTR_LITERAL = 0x0040  # const in C# — value baked in metadata, no storage


# MonoClassRuntimeInfo
class MonoClassRuntimeInfoOffsets:
    MAX_DOMAIN = 0x00  # uint16 max_domain
    VTABLES = 0x08  # MonoVTable* domain_vtables[]; indexed by domain_id


# MonoVTable struct offsets
class MonoVTableOffsets:
    KLASS = 0x00  # MonoClass*
    DOMAIN = 0x10  # MonoDomain*
    STATIC_FIELD_DATA = 0x18  # void* (static fields block)


# MonoObject header
class MonoObjectOffsets:
    VTABLE = 0x00  # MonoVTable*
    SYNC = 0x08  # MonoThreadsSync*
    HEADER_SIZE = 0x10  # Total header before instance fields


# MonoArray layout
class MonoArrayOffsets:
    BOUNDS = 0x10  # MonoArrayBounds*
    MAX_LENGTH = 0x18  # uintptr_t max_length
    DATA = 0x20  # element data starts here


# MonoString layout
class MonoStringOffsets:
    LENGTH = 0x10  # int32 length
    CHARS = 0x14  # char data (UTF-16LE)


# C# List<T> layout (instance fields after MonoObject header)
class CSharpListOffsets:
    ITEMS = 0x10  # T[] _items (MonoArray pointer)
    SIZE = 0x18  # int _size (actual count)
    VERSION = 0x1C  # int _version


# C# Dictionary<K,V> layout (Mono implementation, instance fields after header)
class CSharpDictOffsets:
    TABLE = 0x10  # int[] table (hash bucket indices)
    LINK_SLOTS = 0x18  # Link[] linkSlots (hash chain links)
    KEY_SLOTS = 0x20  # K[] keySlots
    VALUE_SLOTS = 0x28  # V[] valueSlots
    TOUCHED_SLOTS = 0x30  # int touchedSlots (next free slot index)
    EMPTY_SLOT = 0x34  # int emptySlot (-1 if none)
    COUNT = 0x38  # int count (actual number of entries)
    THRESHOLD = 0x3C  # int threshold
    HCP = 0x40  # IEqualityComparer<K> hcp
    GENERATION = 0x50  # int generation


# Mono type enum (subset)
class MonoTypeEnum:
    VOID = 0x01
    BOOLEAN = 0x02
    CHAR = 0x03
    I1 = 0x04
    U1 = 0x05
    I2 = 0x06
    U2 = 0x07
    I4 = 0x08
    U4 = 0x09
    I8 = 0x0A
    U8 = 0x0B
    R4 = 0x0C
    R8 = 0x0D
    STRING = 0x0E
    PTR = 0x0F
    VALUETYPE = 0x11
    CLASS = 0x12
    ARRAY = 0x14
    GENERICINST = 0x15
    SZARRAY = 0x1D
    OBJECT = 0x1C

    @staticmethod
    def name(t: int) -> str:
        _names = {
            0x01: "void",
            0x02: "bool",
            0x03: "char",
            0x04: "sbyte",
            0x05: "byte",
            0x06: "short",
            0x07: "ushort",
            0x08: "int",
            0x09: "uint",
            0x0A: "long",
            0x0B: "ulong",
            0x0C: "float",
            0x0D: "double",
            0x0E: "string",
            0x0F: "ptr",
            0x11: "valuetype",
            0x12: "class",
            0x14: "array",
            0x15: "genericinst",
            0x1C: "object",
            0x1D: "szarray",
        }
        return _names.get(t, f"0x{t:02x}")

    @staticmethod
    def size(t: int) -> int:
        """Size in bytes for primitive types (64-bit)."""
        _sizes = {
            0x02: 1,
            0x03: 2,
            0x04: 1,
            0x05: 1,
            0x06: 2,
            0x07: 2,
            0x08: 4,
            0x09: 4,
            0x0A: 8,
            0x0B: 8,
            0x0C: 4,
            0x0D: 8,
        }
        return _sizes.get(t, 8)  # pointers/refs are 8 bytes on 64-bit


# ---------------------------------------------------------------------------
# Data classes for resolved Mono metadata
# ---------------------------------------------------------------------------


@dataclass
class MonoFieldInfo:
    """Resolved field metadata from a MonoClass."""

    name: str
    offset: int  # Byte offset from object start (instance) or static data block
    type_code: int  # MonoType enum value
    type_name: str  # Human-readable type name
    is_static: bool  # Whether this is a static field
    is_literal: bool  # Whether this is a const/literal field (no runtime storage)
    attrs: int  # Raw field attribute flags
    field_addr: int  # Address of the MonoClassField struct

    def __repr__(self) -> str:
        static = " static" if self.is_static else ""
        literal = " CONST" if self.is_literal else ""
        return f"Field({self.name}:{static}{literal} {self.type_name} @ +0x{self.offset:x})"


@dataclass
class MonoClassInfo:
    """Resolved class metadata."""

    address: int  # Address of the MonoClass struct
    name: str
    namespace: str
    parent_addr: int  # MonoClass* of parent
    image_addr: int  # MonoImage*
    field_count: int
    fields: list[MonoFieldInfo] = field(default_factory=list)
    vtable_addr: int = 0  # MonoVTable* (resolved on demand)
    static_data_addr: int = 0  # Static field data block

    def __repr__(self) -> str:
        ns = f"{self.namespace}." if self.namespace else ""
        return (
            f"MonoClass({ns}{self.name} @ 0x{self.address:x}, "
            f"{self.field_count} fields, vtable=0x{self.vtable_addr:x})"
        )

    def get_field(self, name: str) -> Optional[MonoFieldInfo]:
        """Find a field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class MonoImageInfo:
    """Resolved image metadata."""

    address: int
    name: str
    filename: str
    assembly_addr: int

    def __repr__(self) -> str:
        return f"MonoImage({self.name} @ 0x{self.address:x})"


# ---------------------------------------------------------------------------
# MonoExternal — the main external reader
# ---------------------------------------------------------------------------


class MonoExternal:
    """Walk Mono's internal data structures via process_vm_readv.

    This replaces the need to call Mono API functions. All data is read
    by following pointer chains through the process's memory.

    Args:
        pm: ProcessMemory instance for the target process.
        mono_base: Base address of mono.dll in the target process.
    """

    def __init__(self, pm, mono_base: int):
        from tools.memory_reader import ProcessMemory

        self.pm: ProcessMemory = pm
        self.mono_base = mono_base

        # Cached state
        self._root_domain: int = 0
        self._domain_id: int = -1
        self._images: dict[str, MonoImageInfo] = {}  # name -> info
        self._classes: dict[str, MonoClassInfo] = {}  # "ns.name" -> info
        self._attached = False

    # -- String reading helpers --

    def _read_cstring(self, addr: int, max_len: int = 256) -> Optional[str]:
        """Read a null-terminated C string from the target process."""
        if addr == 0:
            return None
        data = self.pm.read_bytes(addr, max_len)
        if data is None:
            return None
        null_idx = data.find(b"\x00")
        if null_idx >= 0:
            data = data[:null_idx]
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _read_mono_string(self, addr: int) -> Optional[str]:
        """Read a Mono/C# string (MonoString object)."""
        if addr == 0:
            return None
        length = self.pm.read_int32(addr + MonoStringOffsets.LENGTH)
        if length is None or length < 0 or length > 4096:
            return None
        if length == 0:
            return ""
        data = self.pm.read_bytes(addr + MonoStringOffsets.CHARS, length * 2)
        if data is None:
            return None
        try:
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            return None

    # -- Low-level Mono structure reading --

    def _read_root_domain(self) -> int:
        """Read the root domain pointer from mono.dll's .data section."""
        addr = self.mono_base + MonoGlobals.ROOT_DOMAIN_RVA
        domain = self.pm.read_pointer(addr)
        if domain is None or domain == 0:
            logger.error("Failed to read root_domain at 0x%x", addr)
            return 0
        logger.info("Root domain: 0x%x (from 0x%x)", domain, addr)
        return domain

    def _read_domain_id(self) -> int:
        """Read the domain ID from the root domain."""
        if self._root_domain == 0:
            return -1
        did = self.pm.read_int32(self._root_domain + MonoDomainOffsets.DOMAIN_ID)
        if did is None:
            return -1
        logger.info("Domain ID: %d", did)
        return did

    def _iter_domain_assemblies(self):
        """Iterate over the domain's loaded assemblies (GSList traversal).

        Yields (assembly_addr, image_addr, image_name) tuples.
        """
        if self._root_domain == 0:
            return

        node = self.pm.read_pointer(
            self._root_domain + MonoDomainOffsets.DOMAIN_ASSEMBLIES
        )

        count = 0
        while node and node != 0 and count < 200:
            count += 1
            # GSList: data at +0, next at +8
            assembly = self.pm.read_pointer(node + GSListOffsets.DATA)
            if assembly and assembly != 0:
                image = self.pm.read_pointer(assembly + MonoAssemblyOffsets.IMAGE)
                if image and image != 0:
                    name_ptr = self.pm.read_pointer(image + MonoImageOffsets.NAME)
                    name = self._read_cstring(name_ptr) if name_ptr else None

                    filename_ptr = self.pm.read_pointer(
                        image + MonoImageOffsets.NAME_FILENAME
                    )
                    filename = (
                        self._read_cstring(filename_ptr) if filename_ptr else None
                    )

                    yield assembly, image, name or "(unknown)", filename or ""

            node = self.pm.read_pointer(node + GSListOffsets.NEXT)

    def _read_class_at(self, klass_addr: int) -> Optional[MonoClassInfo]:
        """Read MonoClass metadata from an address."""
        if klass_addr == 0:
            return None

        # Read name and namespace
        name_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAME)
        ns_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAMESPACE)

        name = self._read_cstring(name_ptr) if name_ptr else None
        namespace = self._read_cstring(ns_ptr) if ns_ptr else None

        if name is None:
            return None

        parent = self.pm.read_pointer(klass_addr + MonoClassOffsets.PARENT) or 0
        image = self.pm.read_pointer(klass_addr + MonoClassOffsets.IMAGE) or 0
        field_count = (
            self.pm.read_uint32(klass_addr + MonoClassOffsets.FIELD_COUNT) or 0
        )

        return MonoClassInfo(
            address=klass_addr,
            name=name,
            namespace=namespace or "",
            parent_addr=parent,
            image_addr=image,
            field_count=field_count,
        )

    def _read_class_fields(self, klass: MonoClassInfo) -> list[MonoFieldInfo]:
        """Read all fields of a MonoClass."""
        fields = []
        if klass.field_count == 0:
            return fields

        fields_ptr = self.pm.read_pointer(klass.address + MonoClassOffsets.FIELDS)
        if fields_ptr is None or fields_ptr == 0:
            return fields

        for i in range(klass.field_count):
            field_addr = fields_ptr + i * MonoClassFieldOffsets.SIZEOF

            # Read field name
            name_ptr = self.pm.read_pointer(field_addr + MonoClassFieldOffsets.NAME)
            name = self._read_cstring(name_ptr) if name_ptr else f"field_{i}"

            # Read field offset
            offset = self.pm.read_uint32(field_addr + MonoClassFieldOffsets.OFFSET)
            if offset is None:
                offset = 0

            # Read type info
            type_ptr = self.pm.read_pointer(field_addr + MonoClassFieldOffsets.TYPE)
            type_code = 0
            is_static = False
            is_literal = False
            field_attrs = 0
            if type_ptr:
                tc = self.pm.read_int8(type_ptr + MonoTypeOffsets.TYPE)
                if tc is not None:
                    type_code = tc & 0xFF
                # Check static flag from type attrs
                attrs = self.pm.read_uint16(type_ptr + MonoTypeOffsets.ATTRS)
                if attrs is not None:
                    field_attrs = attrs
                    is_static = bool(attrs & MonoTypeOffsets.FIELD_ATTR_STATIC)
                    is_literal = bool(attrs & MonoTypeOffsets.FIELD_ATTR_LITERAL)

            fields.append(
                MonoFieldInfo(
                    name=name or f"field_{i}",
                    offset=offset,
                    type_code=type_code,
                    type_name=MonoTypeEnum.name(type_code),
                    is_static=is_static,
                    is_literal=is_literal,
                    attrs=field_attrs,
                    field_addr=field_addr,
                )
            )

        return fields

    def _resolve_vtable(self, klass: MonoClassInfo) -> int:
        """Resolve the MonoVTable for a class in the root domain.

        Uses: klass->runtime_info->domain_vtables[domain_id]
        """
        if self._domain_id < 0:
            return 0

        runtime_info = self.pm.read_pointer(
            klass.address + MonoClassOffsets.RUNTIME_INFO
        )
        if runtime_info is None or runtime_info == 0:
            logger.debug(
                "No runtime_info for %s (klass=0x%x, ri_addr=0x%x)",
                klass.name,
                klass.address,
                klass.address + MonoClassOffsets.RUNTIME_INFO,
            )
            return 0

        # Check max_domain
        max_domain = self.pm.read_uint16(
            runtime_info + MonoClassRuntimeInfoOffsets.MAX_DOMAIN
        )
        if max_domain is None or max_domain < self._domain_id:
            logger.debug(
                "runtime_info.max_domain=%s < domain_id=%d for %s",
                max_domain,
                self._domain_id,
                klass.name,
            )
            return 0

        # Read vtable pointer from domain_vtables[domain_id]
        vtable = self.pm.read_pointer(
            runtime_info + MonoClassRuntimeInfoOffsets.VTABLES + self._domain_id * 8
        )
        if vtable is None or vtable == 0:
            logger.debug("No vtable for %s in domain %d", klass.name, self._domain_id)
            return 0

        return vtable

    def _resolve_static_data(self, klass: MonoClassInfo) -> int:
        """Get the static field data pointer for a class."""
        if klass.vtable_addr == 0:
            klass.vtable_addr = self._resolve_vtable(klass)
        if klass.vtable_addr == 0:
            return 0

        static_data = self.pm.read_pointer(
            klass.vtable_addr + MonoVTableOffsets.STATIC_FIELD_DATA
        )
        return static_data or 0

    # -- GHashTable walking --

    def _walk_ghashtable(self, ht_addr: int):
        """Walk a GLib GHashTable, yielding (key_ptr, value_ptr) pairs.

        GHashTable layout (from eglib, reverse-engineered):
            +0x00: GHashFunc hash_func
            +0x08: GEqualFunc key_equal_func
            +0x10: GHashNode** table (bucket array)
            +0x18: int table_size (number of buckets)

        GHashNode layout:
            +0x00: void* key
            +0x08: void* value
            +0x10: GHashNode* next
        """
        if ht_addr == 0:
            return

        table_ptr = self.pm.read_pointer(ht_addr + 0x10)
        table_size = self.pm.read_int32(ht_addr + 0x18)

        if not table_ptr or not table_size or table_size <= 0:
            return

        max_chain = 10000  # Guard against corrupted circular chains
        for bucket_idx in range(min(table_size, 100000)):
            node = self.pm.read_pointer(table_ptr + bucket_idx * 8)
            depth = 0
            while node and node != 0 and depth < max_chain:
                key = self.pm.read_pointer(node + 0x00)
                value = self.pm.read_pointer(node + 0x08)
                if key:
                    yield key, value or 0
                node = self.pm.read_pointer(node + 0x10)
                depth += 1
            if depth >= max_chain:
                logger.warning(
                    "GHashTable bucket %d chain exceeded %d — possible corruption",
                    bucket_idx,
                    max_chain,
                )

    # -- Image class enumeration --

    def _get_image_typedef_count(self, image_addr: int) -> int:
        """Get the number of TypeDef rows in an image's metadata."""
        # MonoTableInfo for MONO_TABLE_TYPEDEF (id=2)
        # table_info_addr = image + (table_id + 0x0C) * 0x10
        table_info = image_addr + (2 + 0x0C) * MonoImageOffsets.TABLE_INFO_SIZE
        rows_raw = self.pm.read_uint32(table_info + MonoTableInfoOffsets.ROWS_AND_FLAGS)
        if rows_raw is None:
            return 0
        return rows_raw & 0xFFFFFF

    def _find_class_via_name_cache(
        self, image_addr: int, namespace: str, class_name: str
    ) -> int:
        """Find a class token via the image's name cache (GHashTable at +0x428).

        The name cache is a two-level GHashTable:
            Level 1: namespace (char*) -> GHashTable
            Level 2: class_name (char*) -> typedef_token (uint32, stored as pointer)

        Returns the typedef token, or 0 if not found.
        """
        cache = self.pm.read_pointer(image_addr + MonoImageOffsets.CLASS_CACHE)
        if not cache:
            return 0

        # Level 1: walk namespace -> inner_table
        for ns_key, ns_value in self._walk_ghashtable(cache):
            ns_str = self._read_cstring(ns_key)
            if ns_str != namespace:
                continue

            # Level 2: walk name -> token
            if ns_value == 0:
                continue
            for name_key, token_value in self._walk_ghashtable(ns_value):
                name_str = self._read_cstring(name_key)
                if name_str == class_name:
                    # token_value is the typedef token (stored as a pointer-sized int)
                    return token_value

        return 0

    def _iter_image_class_cache(self, image_addr: int):
        """Iterate all loaded MonoClass* from the image's internal hash table.

        The class cache is a MonoInternalHashTable at MonoImage + 0x3D0.
        Layout:
            +0x18: uint32 num_buckets
            +0x1C: uint32 num_entries
            +0x20: MonoClass** table (bucket array)

        Each bucket is a singly-linked list of MonoClass* chained via
        MonoClass + 0x108 (next_class_cache). The hash key is the
        type_token at MonoClass + 0x60.

        Yields MonoClass* addresses for all loaded classes in this image.
        """
        num_buckets = self.pm.read_uint32(
            image_addr + MonoImageOffsets.CLASS_CACHE_HT_NUM_BUCKETS
        )
        if num_buckets is None or num_buckets == 0:
            logger.warning("Class cache hash table has 0 buckets")
            return

        bucket_array = self.pm.read_pointer(
            image_addr + MonoImageOffsets.CLASS_CACHE_HT_TABLE
        )
        if not bucket_array:
            logger.warning("Class cache bucket array is NULL")
            return

        num_entries = (
            self.pm.read_uint32(
                image_addr + MonoImageOffsets.CLASS_CACHE_HT_NUM_ENTRIES
            )
            or 0
        )
        logger.info(
            "Walking class cache: %d buckets, %d entries, table=0x%x",
            num_buckets,
            num_entries,
            bucket_array,
        )

        yielded = 0
        for i in range(num_buckets):
            klass = self.pm.read_pointer(bucket_array + i * 8)
            depth = 0
            while klass and klass != 0 and depth < 500:
                depth += 1
                yielded += 1
                yield klass
                # Follow chain link
                klass = self.pm.read_pointer(klass + MonoClassOffsets.NEXT_CLASS_CACHE)

        logger.info("Class cache walk yielded %d classes", yielded)

    def _find_class_in_cache(
        self, image_addr: int, namespace: str, class_name: str
    ) -> int:
        """Find a MonoClass* by name in the image's internal hash table.

        Returns the MonoClass* address, or 0 if not found.
        """
        for klass_addr in self._iter_image_class_cache(image_addr):
            # Fast check: read name pointer first
            name_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAME)
            if not name_ptr:
                continue
            name = self._read_cstring(name_ptr)
            if name != class_name:
                continue

            # Check namespace
            ns_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAMESPACE)
            ns = self._read_cstring(ns_ptr) if ns_ptr else ""
            if (ns or "") == namespace:
                return klass_addr

        return 0

    def _find_classes_in_cache(
        self, image_addr: int, wanted: set[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Find multiple classes by name in the image's internal hash table.

        Args:
            image_addr: The MonoImage* address
            wanted: Set of (namespace, class_name) tuples to find

        Returns:
            Dict of (namespace, class_name) -> MonoClass* address for found classes
        """
        results: dict[tuple[str, str], int] = {}
        remaining = set(wanted)

        for klass_addr in self._iter_image_class_cache(image_addr):
            if not remaining:
                break

            name_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAME)
            if not name_ptr:
                continue
            name = self._read_cstring(name_ptr)
            if name is None:
                continue

            ns_ptr = self.pm.read_pointer(klass_addr + MonoClassOffsets.NAMESPACE)
            ns = self._read_cstring(ns_ptr) if ns_ptr else ""
            ns = ns or ""

            key = (ns, name)
            if key in remaining:
                results[key] = klass_addr
                remaining.discard(key)

        return results

    # -- Public API --

    def attach(self) -> bool:
        """Initialize: read root domain, enumerate assemblies.

        Returns True if successfully attached to the Mono runtime.
        """
        self._root_domain = self._read_root_domain()
        if self._root_domain == 0:
            return False

        self._domain_id = self._read_domain_id()

        # Enumerate loaded assemblies/images
        logger.info("Enumerating loaded assemblies...")
        self._images.clear()
        for assembly, image, name, filename in self._iter_domain_assemblies():
            info = MonoImageInfo(
                address=image,
                name=name,
                filename=filename,
                assembly_addr=assembly,
            )
            self._images[name] = info
            logger.info("  Assembly: %s (image=0x%x)", name, image)

        logger.info("Found %d assemblies", len(self._images))
        self._attached = bool(self._images)
        return self._attached

    @property
    def is_attached(self) -> bool:
        return self._attached

    def list_assemblies(self) -> list[str]:
        """Return names of all loaded assemblies."""
        return list(self._images.keys())

    def get_image(self, name: str) -> Optional[MonoImageInfo]:
        """Get a loaded image by name (e.g. 'Assembly-CSharp')."""
        return self._images.get(name)

    def find_class(
        self, image_name: str, namespace: str, class_name: str
    ) -> Optional[MonoClassInfo]:
        """Find a MonoClass by image name, namespace, and class name.

        Walks the image's MonoInternalHashTable class cache at +0x3D0
        to find loaded classes by name. Results are cached.

        Args:
            image_name: Assembly image name (e.g. "Assembly-CSharp")
            namespace: C# namespace (e.g. "" for global)
            class_name: C# class name (e.g. "PlayerStatsManager")

        Returns:
            MonoClassInfo with resolved fields and vtable, or None.
        """
        cache_key = f"{namespace}.{class_name}" if namespace else class_name
        if cache_key in self._classes:
            return self._classes[cache_key]

        image = self._images.get(image_name)
        if image is None:
            logger.error("Image '%s' not found", image_name)
            return None

        logger.info(
            "Searching class cache in %s for %s.%s...",
            image_name,
            namespace,
            class_name,
        )

        klass_addr = self._find_class_in_cache(image.address, namespace, class_name)
        if klass_addr == 0:
            logger.warning(
                "Class %s.%s not found in %s class cache",
                namespace,
                class_name,
                image_name,
            )
            return None

        logger.info("Found %s at 0x%x", class_name, klass_addr)
        klass = self._read_class_at(klass_addr)
        if klass is None:
            return None

        # Read fields
        klass.fields = self._read_class_fields(klass)

        # Resolve vtable and static data
        klass.vtable_addr = self._resolve_vtable(klass)
        if klass.vtable_addr:
            klass.static_data_addr = self._resolve_static_data(klass)

        self._classes[cache_key] = klass
        return klass

    def find_classes_batch(
        self, image_name: str, class_specs: list[tuple[str, str]]
    ) -> dict[str, Optional[MonoClassInfo]]:
        """Find multiple classes in a single pass over the class cache.

        Walks the MonoInternalHashTable once and matches all requested
        classes. Much more efficient than calling find_class() for each one.

        Args:
            image_name: Assembly image name
            class_specs: List of (namespace, class_name) tuples

        Returns:
            Dict of "namespace.class_name" -> MonoClassInfo
        """
        image = self._images.get(image_name)
        if image is None:
            return {}

        # Build lookup set
        wanted = {(ns, cn) for ns, cn in class_specs}
        results: dict[str, Optional[MonoClassInfo]] = {}
        for ns, cn in class_specs:
            key = f"{ns}.{cn}" if ns else cn
            # Check cache first
            if key in self._classes:
                results[key] = self._classes[key]
                wanted.discard((ns, cn))

        if not wanted:
            return results

        logger.info(
            "Batch searching %d classes in %s class cache",
            len(wanted),
            image_name,
        )

        found = self._find_classes_in_cache(image.address, wanted)

        for (ns, cn), klass_addr in found.items():
            key = f"{ns}.{cn}" if ns else cn
            logger.info("Found %s at 0x%x", key, klass_addr)

            klass = self._read_class_at(klass_addr)
            if klass is None:
                results[key] = None
                continue

            klass.fields = self._read_class_fields(klass)
            klass.vtable_addr = self._resolve_vtable(klass)
            if klass.vtable_addr:
                klass.static_data_addr = self._resolve_static_data(klass)

            self._classes[key] = klass
            results[key] = klass
            wanted.discard((ns, cn))

        # Mark not-found classes
        for ns, cn in wanted:
            key = f"{ns}.{cn}" if ns else cn
            results[key] = None
            logger.warning(
                "Class %s.%s not found in %s class cache", ns, cn, image_name
            )

        return results

    # -- Reading field values --

    def _ensure_static_data(self, klass: MonoClassInfo) -> bool:
        """Ensure the class has its static data resolved (lazy vtable retry).

        Returns True if static_data_addr is available, False otherwise.
        """
        if klass.static_data_addr != 0:
            return True
        # Retry vtable resolution — class may have been initialized since
        # we first looked (e.g., player enters a level)
        klass.vtable_addr = self._resolve_vtable(klass)
        if klass.vtable_addr:
            klass.static_data_addr = self._resolve_static_data(klass)
        return klass.static_data_addr != 0

    def read_static_field_value(
        self, klass: MonoClassInfo, field_name: str
    ) -> Optional[int]:
        """Read a static field's raw bytes as an integer.

        For pointer-sized fields (class/object refs), returns the pointer value.
        For int32, returns the int value.
        """
        field = klass.get_field(field_name)
        if field is None:
            logger.warning("Field '%s' not found in %s", field_name, klass.name)
            return None

        if not self._ensure_static_data(klass):
            logger.debug("No static data for %s", klass.name)
            return None

        addr = klass.static_data_addr + field.offset
        size = MonoTypeEnum.size(field.type_code)

        if size == 1:
            return self.pm.read_uint8(addr)
        elif size == 2:
            return self.pm.read_uint16(addr)
        elif size == 4:
            return self.pm.read_uint32(addr)
        else:
            return self.pm.read_uint64(addr)

    def read_static_field_ptr(
        self, klass: MonoClassInfo, field_name: str
    ) -> Optional[int]:
        """Read a static field that's a pointer (class/object reference)."""
        return self.read_static_field_value(klass, field_name)

    def read_static_field_int(
        self, klass: MonoClassInfo, field_name: str
    ) -> Optional[int]:
        """Read a static int32 field."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        if not self._ensure_static_data(klass):
            return None
        return self.pm.read_int32(klass.static_data_addr + field.offset)

    def read_static_field_bool(
        self, klass: MonoClassInfo, field_name: str
    ) -> Optional[bool]:
        """Read a static bool field."""
        val = self.read_static_field_value(klass, field_name)
        return bool(val) if val is not None else None

    def read_static_field_float(
        self, klass: MonoClassInfo, field_name: str
    ) -> Optional[float]:
        """Read a static float field."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        if not self._ensure_static_data(klass):
            return None
        return self.pm.read_float(klass.static_data_addr + field.offset)

    # -- Instance field reading (from object pointers) --

    def read_instance_field_int(
        self, obj_addr: int, klass: MonoClassInfo, field_name: str
    ) -> Optional[int]:
        """Read an int32 instance field from an object."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        return self.pm.read_int32(obj_addr + field.offset)

    def read_instance_field_float(
        self, obj_addr: int, klass: MonoClassInfo, field_name: str
    ) -> Optional[float]:
        """Read a float instance field from an object."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        return self.pm.read_float(obj_addr + field.offset)

    def read_instance_field_bool(
        self, obj_addr: int, klass: MonoClassInfo, field_name: str
    ) -> Optional[bool]:
        """Read a bool instance field from an object."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        val = self.pm.read_uint8(obj_addr + field.offset)
        return bool(val) if val is not None else None

    def read_instance_field_ptr(
        self, obj_addr: int, klass: MonoClassInfo, field_name: str
    ) -> Optional[int]:
        """Read a pointer/reference instance field from an object."""
        field = klass.get_field(field_name)
        if field is None:
            return None
        return self.pm.read_pointer(obj_addr + field.offset)

    # -- Debug / exploration --

    def dump_class(self, klass: MonoClassInfo) -> str:
        """Pretty-print a class with all its fields."""
        lines = [repr(klass)]
        if klass.static_data_addr:
            lines.append(f"  static_data: 0x{klass.static_data_addr:x}")
        for f in klass.fields:
            static = " [STATIC]" if f.is_static else ""
            lines.append(f"  +0x{f.offset:04x}: {f.name} ({f.type_name}){static}")
        return "\n".join(lines)

    def dump_all_images(self) -> str:
        """Pretty-print all loaded assemblies."""
        lines = [f"Loaded assemblies ({len(self._images)}):"]
        for name, info in sorted(self._images.items()):
            num_types = self._get_image_typedef_count(info.address)
            lines.append(f"  {name:40s} image=0x{info.address:x}  types={num_types}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from tools.memory_reader import ProcessMemory, find_wine_pid

    # Find game process
    pid = find_wine_pid("Cuphead.exe")
    if pid is None:
        print("Cuphead.exe not running")
        sys.exit(1)

    pm = ProcessMemory(pid)
    print(f"Attached to PID {pid}")

    # Find mono.dll base
    mono_base = pm.get_module_base("mono.dll")
    if mono_base is None:
        print("mono.dll not found in process maps")
        sys.exit(1)
    print(f"mono.dll base: 0x{mono_base:x}")

    # Create external reader
    mono = MonoExternal(pm, mono_base)
    if not mono.attach():
        print("Failed to attach to Mono runtime")
        sys.exit(1)

    print(f"\n{mono.dump_all_images()}")

    # Find key game classes
    print("\n--- Finding game classes ---")
    classes = mono.find_classes_batch(
        "Assembly-CSharp",
        [
            ("", "PlayerStatsManager"),
            ("", "PlayerManager"),
            ("", "Level"),
            ("", "PlayerData"),
            ("", "SceneLoader"),
        ],
    )

    for key, klass in classes.items():
        if klass is None:
            print(f"\n{key}: NOT FOUND")
        else:
            print(f"\n{mono.dump_class(klass)}")

    # Try reading some game state
    print("\n--- Reading game state ---")

    level = classes.get("Level")
    if level and level.static_data_addr:
        current_field = level.get_field("<Current>k__BackingField")
        if current_field:
            current = mono.read_static_field_ptr(level, "<Current>k__BackingField")
            print(f"Level.Current: 0x{current or 0:x}")
            if current:
                time_field = level.get_field("LevelTime")
                if time_field is None:
                    # LevelTime might be called differently or in parent
                    print("  LevelTime field not found in Level class fields:")
                    for f in level.fields:
                        print(f"    {f}")

    pm_class = classes.get("PlayerManager")
    if pm_class:
        players = mono.read_static_field_ptr(pm_class, "players")
        print(f"PlayerManager.players: 0x{players or 0:x}")

    pd = classes.get("PlayerData")
    if pd:
        in_game = mono.read_static_field_ptr(pd, "inGame")
        print(f"PlayerData.inGame ptr: 0x{in_game or 0:x}")
