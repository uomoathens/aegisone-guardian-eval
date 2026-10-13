from __future__ import annotations

import math
import os
import struct
from datetime import datetime, timezone

VERSION = 1
LENGTH = 48

WINDOW_BYTES = 256 * 1024
HISTOGRAM_BUCKETS = 32

IDX_LOG_SIZE = 0
IDX_HEADER_ENTROPY = 1
IDX_MID_ENTROPY = 2
IDX_TAIL_ENTROPY = 3
IDX_MAX_ENTROPY = 4
IDX_ENTROPY_GAP = 5
IDX_HISTOGRAM = 6
IDX_PRINTABLE_RATIO = 38
IDX_NULL_RATIO = 39
IDX_IS_PE = 40
IDX_SECTION_COUNT = 41
IDX_IS_DLL = 42
IDX_TIMESTAMP_BAD = 43
IDX_IS_64BIT = 44
IDX_IS_GUI_SUBSYSTEM = 45
IDX_RELOCS_STRIPPED = 46
IDX_OVERLAY_RATIO = 47

FEATURE_NAMES = (
    ["log_size", "header_entropy", "mid_entropy", "tail_entropy",
     "max_entropy", "entropy_gap"]
    + [f"hist_{i:02d}" for i in range(HISTOGRAM_BUCKETS)]
    + ["printable_ratio", "null_ratio", "is_pe", "section_count", "is_dll",
       "timestamp_bad", "is_64bit", "is_gui_subsystem", "relocs_stripped",
       "overlay_ratio"]
)
assert len(FEATURE_NAMES) == LENGTH


def _clamp01(value: float) -> float:
    if value != value:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _entropy(freq: list[int], total: int) -> float:
    if total == 0:
        return 0.0
    result = 0.0
    for count in freq:
        if count:
            p = count / total
            result -= p * math.log2(p)
    return result


class _Window:
    __slots__ = ("freq", "total")

    def __init__(self) -> None:
        self.freq = [0] * 256
        self.total = 0

    def entropy(self) -> float:
        return _entropy(self.freq, self.total)

    def bucket(self, index: int) -> float:
        if self.total == 0:
            return 0.0
        start = index * 8
        return sum(self.freq[start:start + 8]) / self.total

    def printable_ratio(self) -> float:
        if self.total == 0:
            return 0.0
        total = sum(self.freq[0x20:0x7F])
        total += self.freq[0x09] + self.freq[0x0A] + self.freq[0x0D]
        return total / self.total

    def null_ratio(self) -> float:
        return 0.0 if self.total == 0 else self.freq[0] / self.total


def _read_window(handle, offset: int, length: int) -> _Window:
    window = _Window()
    if length <= 0:
        return window

    offset = max(0, offset)
    try:
        handle.seek(offset)
    except OSError:
        return window

    remaining = length
    while remaining > 0:
        chunk = handle.read(min(65536, remaining))
        if not chunk:
            break
        for byte in chunk:
            window.freq[byte] += 1
        window.total += len(chunk)
        remaining -= len(chunk)

    return window


def _merge(*windows: _Window) -> _Window:
    merged = _Window()
    seen = set()
    for window in windows:
        if id(window) in seen:
            continue
        seen.add(id(window))
        for i in range(256):
            merged.freq[i] += window.freq[i]
        merged.total += window.total
    return merged


def _read_pe(handle, size: int) -> dict:
    pe = {
        "is_pe": False, "sections": 0, "is_dll": False, "timestamp_bad": False,
        "is_64bit": False, "is_gui": False, "relocs_stripped": False,
        "end_of_last_section": 0,
    }

    try:
        if size < 0x40:
            return pe

        handle.seek(0)
        if handle.read(2) != b"MZ":
            return pe

        handle.seek(0x3C)
        raw = handle.read(4)
        if len(raw) < 4:
            return pe
        pe_offset = struct.unpack("<i", raw)[0]
        if pe_offset <= 0 or pe_offset + 24 > size:
            return pe

        handle.seek(pe_offset)
        coff = handle.read(24)
        if len(coff) < 24 or coff[:4] != b"PE\x00\x00":
            return pe

        machine = struct.unpack_from("<H", coff, 4)[0]
        sections = struct.unpack_from("<H", coff, 6)[0]
        timestamp = struct.unpack_from("<I", coff, 8)[0]
        optional_size = struct.unpack_from("<H", coff, 20)[0]
        characteristics = struct.unpack_from("<H", coff, 22)[0]

        pe["is_pe"] = True
        pe["sections"] = sections
        pe["is_dll"] = bool(characteristics & 0x2000)
        pe["relocs_stripped"] = bool(characteristics & 0x0001)
        pe["is_64bit"] = machine in (0x8664, 0xAA64)

        if timestamp == 0:
            pe["timestamp_bad"] = True
        else:
            try:
                linked = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                low = datetime(1995, 1, 1, tzinfo=timezone.utc)
                high = datetime.now(timezone.utc)
                pe["timestamp_bad"] = linked < low or (linked - high).days > 1
            except (OverflowError, OSError, ValueError):
                pe["timestamp_bad"] = True

        optional_offset = pe_offset + 24
        if optional_size >= 70 and optional_offset + 70 <= size:
            handle.seek(optional_offset + 68)
            raw = handle.read(2)
            if len(raw) == 2:
                pe["is_gui"] = struct.unpack("<H", raw)[0] == 2

        section_table = optional_offset + optional_size
        end = 0
        for i in range(min(sections, 96)):
            entry = section_table + i * 40
            if entry + 40 > size:
                break
            handle.seek(entry + 16)
            raw = handle.read(8)
            if len(raw) < 8:
                break
            size_of_raw, pointer_to_raw = struct.unpack("<II", raw)
            candidate = pointer_to_raw + size_of_raw
            if end < candidate <= size:
                end = candidate

        pe["end_of_last_section"] = end
        return pe

    except (OSError, struct.error):
        return pe


def compute(path: str) -> list[float] | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None

    vector = [0.0] * LENGTH

    try:
        with open(path, "rb") as handle:
            head = _read_window(handle, 0, min(WINDOW_BYTES, size))

            tail = _read_window(handle, size - WINDOW_BYTES, WINDOW_BYTES) if size > WINDOW_BYTES else head

            if size > WINDOW_BYTES * 3:
                mid = _read_window(handle, (size // 2) - (WINDOW_BYTES // 2), WINDOW_BYTES)
            else:
                mid = head

            combined = _merge(head, mid, tail)

            head_e, mid_e, tail_e = head.entropy(), mid.entropy(), tail.entropy()

            vector[IDX_LOG_SIZE] = _clamp01(math.log10(size + 1.0) / 10.0)
            vector[IDX_HEADER_ENTROPY] = _clamp01(head_e / 8.0)
            vector[IDX_MID_ENTROPY] = _clamp01(mid_e / 8.0)
            vector[IDX_TAIL_ENTROPY] = _clamp01(tail_e / 8.0)
            vector[IDX_MAX_ENTROPY] = _clamp01(max(head_e, mid_e, tail_e) / 8.0)
            vector[IDX_ENTROPY_GAP] = _clamp01((tail_e - head_e + 8.0) / 16.0)

            for bucket in range(HISTOGRAM_BUCKETS):
                vector[IDX_HISTOGRAM + bucket] = _clamp01(combined.bucket(bucket))

            vector[IDX_PRINTABLE_RATIO] = _clamp01(combined.printable_ratio())
            vector[IDX_NULL_RATIO] = _clamp01(combined.null_ratio())

            pe = _read_pe(handle, size)

            vector[IDX_IS_PE] = 1.0 if pe["is_pe"] else 0.0
            vector[IDX_SECTION_COUNT] = _clamp01(pe["sections"] / 16.0)
            vector[IDX_IS_DLL] = 1.0 if pe["is_dll"] else 0.0
            vector[IDX_TIMESTAMP_BAD] = 1.0 if pe["timestamp_bad"] else 0.0
            vector[IDX_IS_64BIT] = 1.0 if pe["is_64bit"] else 0.0
            vector[IDX_IS_GUI_SUBSYSTEM] = 1.0 if pe["is_gui"] else 0.0
            vector[IDX_RELOCS_STRIPPED] = 1.0 if pe["relocs_stripped"] else 0.0

            if pe["is_pe"] and pe["end_of_last_section"] > 0 and size > 0:
                overlay = (size - pe["end_of_last_section"]) / size
                vector[IDX_OVERLAY_RATIO] = _clamp01(overlay)

    except OSError:
        return None

    return vector
