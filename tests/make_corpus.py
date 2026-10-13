import os
import random
import struct
import sys
import time

W = 256 * 1024


def pe(timestamp=0x60000000, machine=0x8664, characteristics=0x0022, sections=None,
       optional_size=240, subsystem=2, e_lfanew=0x80, body=4096, overlay=b"", truncate=None,
       section_count_field=None, fill_end=None):
    if sections is None:
        sections = [(0x400, 0x1000), (0x1400, 0x800)]
    head = bytearray(b"MZ" + b"\x90" * 0x3A + struct.pack("<i", e_lfanew))
    head += b"\x00" * max(0, e_lfanew - len(head))
    count = len(sections) if section_count_field is None else section_count_field
    coff = b"PE\x00\x00" + struct.pack("<HHIIIHH", machine, count, timestamp, 0, 0, optional_size, characteristics)
    optional = bytearray(optional_size)
    if optional_size >= 70:
        struct.pack_into("<H", optional, 68, subsystem)
    table = b""
    for pointer, size in sections:
        table += b".text\x00\x00\x00" + struct.pack("<IIII", size, 0x1000, size, pointer) + b"\x00" * 16
    data = bytes(head) + coff + bytes(optional) + table
    end = fill_end if fill_end is not None else max([p + s for p, s in sections] + [len(data)])
    rng = random.Random(len(data) + body)
    filler = bytes(rng.getrandbits(8) for _ in range(max(0, end - len(data)))) if body else b"\x00" * max(0, end - len(data))
    data += filler + overlay
    if truncate is not None:
        data = data[:truncate]
    return data


def main(out):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(7)
    now = int(time.time())
    cases = {
        "empty.bin": b"",
        "one_byte.bin": b"A",
        "mz_tiny.bin": b"MZ" + b"\x00" * 0x3D,
        "text.txt": ("hello world\r\n\t" * 5000).encode(),
        "zeros_w.bin": b"\x00" * W,
        "random_w_plus_1.bin": bytes(rng.getrandbits(8) for _ in range(W + 1)),
        "random_3w.bin": bytes(rng.getrandbits(8) for _ in range(3 * W)),
        "random_3w_plus_1.bin": bytes(rng.getrandbits(8) for _ in range(3 * W + 1)),
        "stub_plus_payload_600k.bin": b"\x00" * (400 * 1024) + bytes(rng.getrandbits(8) for _ in range(200 * 1024)),
        "pe_basic.exe": pe(),
        "pe_dll32.dll": pe(machine=0x14C, characteristics=0x2103, subsystem=3),
        "pe_arm64.exe": pe(machine=0xAA64),
        "pe_ts_zero.exe": pe(timestamp=0),
        "pe_ts_1990.exe": pe(timestamp=631152000),
        "pe_ts_future_1_5d.exe": pe(timestamp=now + 36 * 3600),
        "pe_ts_future_3d.exe": pe(timestamp=now + 3 * 86400),
        "pe_ts_max.exe": pe(timestamp=0xFFFFFFFF),
        "pe_overlay.exe": pe(overlay=bytes(rng.getrandbits(8) for _ in range(300 * 1024))),
        "pe_small_optional.exe": pe(optional_size=40),
        "pe_negative_lfanew.exe": pe(e_lfanew=0x80)[:0x3C] + struct.pack("<i", -5) + pe()[0x40:],
        "pe_lfanew_beyond.exe": pe()[:0x3C] + struct.pack("<i", 10_000_000) + pe()[0x40:],
        "pe_truncated_coff.exe": pe(truncate=0x80 + 10),
        "pe_truncated_table.exe": pe(truncate=0x80 + 24 + 240 + 50),
        "pe_many_sections.exe": pe(sections=[(0x400 + i * 0x200, 0x200) for i in range(120)]),
        "pe_section_beyond_eof.exe": pe(sections=[(0x400, 0x1000), (0x10_000_000, 0x1000)], fill_end=0x1400),
        "pe_bad_signature.exe": pe().replace(b"PE\x00\x00", b"PX\x00\x00", 1),
        "pe_big_mid.exe": pe(overlay=bytes(rng.getrandbits(8) for _ in range(900 * 1024))),
    }
    for name, data in cases.items():
        with open(os.path.join(out, name), "wb") as handle:
            handle.write(data)
    print(f"wrote {len(cases)} synthetic files to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
