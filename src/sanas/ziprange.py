"""Random access into the beintelli_v1 ZIP over HTTP range requests.

The dataset ships as one 73.5 GB ZIP. Both mirrors answer HTTP Range requests
with 206, and every member is individually deflate-compressed with its own
local header, so any subset can be pulled without downloading the archive.

Verified 2026-08-09:
  - Zenodo record 20559664 lists exactly one file, 73,515,214,805 bytes,
    md5 74924b5be59e706a5a89affba48f6b87.
  - The HuggingFace mirror serves a byte-identical file (same length) and was
    measured ~8.6x faster than Zenodo from a residential connection.
  - The archive is ZIP64: 298,694 members, central directory 44,954,940 bytes
    at offset 73,470,259,767.

Stdlib only, no third-party imports, so this file can be vendored into a
Kaggle script kernel unchanged.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass

MIRRORS = {
    "hf": (
        "https://huggingface.co/datasets/evgenygorelik96/"
        "multiview_incabin_dataset/resolve/main/beintelli_v1.zip"
    ),
    "zenodo": "https://zenodo.org/api/records/20559664/files/beintelli_v1.zip/content",
}
DEFAULT_MIRROR = "hf"

ARCHIVE_NAME = "beintelli_v1.zip"
ARCHIVE_SIZE = 73_515_214_805
ARCHIVE_MD5 = "74924b5be59e706a5a89affba48f6b87"
ROOT_PREFIX = "beintelli_v1/"

# A local file header is 30 bytes + filename + extra field. The central
# directory does not record the *local* extra length, so a run's end is an
# estimate. Padding by the legal maximum (65535) would waste ~25% of the
# transfer on a stride-10 selection, so pad small and let the reader re-fetch
# exactly when a member turns out to overrun its run. Observed local extra
# length in this archive is 32 bytes.
_LOCAL_EXTRA_PAD = 1024

_USER_AGENT = "sanas-occupancy/0.1 (+research; contact via project repo)"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def http_get(
    url: str,
    start: int | None = None,
    end: int | None = None,
    retries: int = 5,
    timeout: int = 120,
    backoff: float = 2.0,
) -> bytes:
    """GET url, optionally a byte range (inclusive bounds), with retries."""
    headers = {"User-Agent": _USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if start is not None and resp.status != 206:
                    raise OSError(
                        f"server ignored Range (status {resp.status}); "
                        "mirror does not support partial download"
                    )
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff**attempt)
    raise OSError(f"GET failed after {retries} attempts: {url} ({last})")


def remote_size(url: str, timeout: int = 60) -> int:
    """Content-Length of the remote archive."""
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers["Content-Length"])


# --------------------------------------------------------------------------
# Central directory
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    """One entry of the ZIP central directory."""

    name: str
    method: int  # 0 = stored, 8 = deflate
    size: int  # uncompressed
    csize: int  # compressed
    offset: int  # local header offset
    crc: int

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


def _find_eocd(tail: bytes) -> tuple[int, int, int]:
    """Return (entry_count, cd_size, cd_offset) from the ZIP tail."""
    i = tail.rfind(b"PK\x05\x06")
    if i == -1:
        raise ValueError("end-of-central-directory record not found in tail")
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i : i + 22])

    j = tail.rfind(b"PK\x06\x06")
    if j != -1:
        z = struct.unpack("<IQHHIIQQQQ", tail[j : j + 56])
        n, cd_size, cd_off = z[7], z[8], z[9]
    if cd_off == 0xFFFFFFFF:
        raise ValueError("ZIP64 locator required but not found")
    return n, cd_size, cd_off


def parse_central_directory(blob: bytes) -> list[Member]:
    """Parse a raw central directory into Member records (ZIP64 aware)."""
    members: list[Member] = []
    off = 0
    end = len(blob)
    while off + 46 <= end:
        if blob[off : off + 4] != b"PK\x01\x02":
            break
        (
            _sig,
            _vmb,
            _vnb,
            _flags,
            method,
            _mtime,
            _mdate,
            crc,
            csize,
            usize,
            nlen,
            elen,
            clen,
            _dstart,
            _iattr,
            _eattr,
            lhoff,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", blob[off : off + 46])

        name = blob[off + 46 : off + 46 + nlen].decode("utf-8", "replace")
        extra = blob[off + 46 + nlen : off + 46 + nlen + elen]

        # ZIP64 extended information overrides the 0xFFFFFFFF sentinels.
        eo = 0
        while eo + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[eo : eo + 4])
            body = extra[eo + 4 : eo + 4 + hsz]
            if hid == 0x0001:
                bo = 0
                if usize == 0xFFFFFFFF and bo + 8 <= len(body):
                    usize = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
                if csize == 0xFFFFFFFF and bo + 8 <= len(body):
                    csize = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
                if lhoff == 0xFFFFFFFF and bo + 8 <= len(body):
                    lhoff = struct.unpack("<Q", body[bo : bo + 8])[0]
                    bo += 8
            eo += 4 + hsz

        members.append(Member(name, method, usize, csize, lhoff, crc))
        off += 46 + nlen + elen + clen
    return members


def build_index(url: str, verbose: bool = True) -> dict:
    """Fetch and parse the central directory. Downloads ~45 MB, not the archive."""
    total = remote_size(url)
    if verbose:
        print(f"archive size: {total:,} bytes", flush=True)

    tail = http_get(url, total - 65_536, total - 1)
    n_entries, cd_size, cd_off = _find_eocd(tail)
    if verbose:
        print(
            f"central directory: {n_entries:,} entries, "
            f"{cd_size:,} bytes at offset {cd_off:,}",
            flush=True,
        )

    blob = http_get(url, cd_off, cd_off + cd_size - 1)
    members = parse_central_directory(blob)
    if len(members) != n_entries:
        print(
            f"WARNING: parsed {len(members)} entries, EOCD claims {n_entries}",
            file=sys.stderr,
        )

    return {
        "archive": ARCHIVE_NAME,
        "archive_size": total,
        "archive_md5": ARCHIVE_MD5 if total == ARCHIVE_SIZE else None,
        "source_url": url,
        "entry_count": len(members),
        "members": [asdict(m) for m in members],
    }


def save_index(path: str, index: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)


def load_index(path: str) -> tuple[dict, list[Member]]:
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    return index, [Member(**m) for m in index["members"]]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def sniff_image_ext(data: bytes) -> str | None:
    """Return the real extension for an image payload, or None if unknown.

    The dataset's depth frames are named ``*.jpg`` but are actually PNG
    (magic 89 50 4E 47). Anything decoding by extension will mislabel them.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    return None


def _run_end(member: Member, archive_size: int) -> int:
    """Exclusive upper bound on the bytes a member can occupy."""
    return min(
        archive_size,
        member.offset
        + 30
        + len(member.name.encode())
        + _LOCAL_EXTRA_PAD
        + member.csize,
    )


def plan_runs(
    members: list[Member],
    archive_size: int,
    gap_merge: int = 1 << 20,
    max_run_bytes: int = 256 << 20,
) -> list[tuple[int, int, list[Member]]]:
    """Group members into contiguous byte runs to keep the request count sane.

    Fetching 130k tiny annotation files one request each is not viable; the
    annotation directories are physically contiguous at the end of the archive
    so they collapse into a handful of runs.
    """
    ordered = sorted((m for m in members if not m.is_dir), key=lambda m: m.offset)
    runs: list[tuple[int, int, list[Member]]] = []
    for m in ordered:
        m_end = _run_end(m, archive_size)
        if runs:
            start, end, group = runs[-1]
            if m.offset - end <= gap_merge and (m_end - start) <= max_run_bytes:
                group.append(m)
                runs[-1] = (start, max(end, m_end), group)
                continue
        runs.append((m.offset, m_end, [m]))
    return runs


def _member_data(buf: bytes, member: Member, base: int, url: str) -> tuple[bytes, int]:
    """Slice one member out of a run buffer and decompress it.

    Falls back to an exact range request for whatever the run buffer did not
    cover, so the run-end estimate never has to be conservative. Returns
    (data, extra_requests).
    """
    extra = 0
    pos = member.offset - base
    if pos < 0 or pos + 30 > len(buf):
        header = http_get(url, member.offset, member.offset + 29)
        extra += 1
    else:
        header = buf[pos : pos + 30]

    sig, _ver, _flg, _meth, _t, _d, _crc, _c, _u, nlen, elen = struct.unpack(
        "<IHHHHHIIIHH", header
    )
    if sig != 0x04034B50:
        raise ValueError(f"bad local header signature for {member.name}")

    data_start = member.offset + 30 + nlen + elen
    lo = data_start - base
    if lo >= 0 and lo + member.csize <= len(buf):
        raw = buf[lo : lo + member.csize]
    else:
        raw = http_get(url, data_start, data_start + member.csize - 1)
        extra += 1
    if len(raw) != member.csize:
        raise ValueError(f"short read for {member.name}")

    data = zlib.decompress(raw, -15) if member.method == 8 else raw
    if member.crc and zlib.crc32(data) & 0xFFFFFFFF != member.crc:
        raise ValueError(f"CRC mismatch for {member.name}")
    return data, extra


def extract_members(
    url: str,
    members: list[Member],
    out_dir: str,
    archive_size: int = ARCHIVE_SIZE,
    strip_prefix: str = ROOT_PREFIX,
    gap_merge: int = 1 << 20,
    max_run_bytes: int = 256 << 20,
    fix_image_extensions: bool = True,
    verbose: bool = True,
) -> dict:
    """Download and inflate the given members into out_dir.

    Returns a summary dict. Members are written at their archive-relative
    path with ``strip_prefix`` removed. When ``fix_image_extensions`` is set,
    a file whose payload disagrees with its extension is written with the
    correct one (this is what handles the depth-frames-are-PNG problem).
    """
    runs = plan_runs(members, archive_size, gap_merge, max_run_bytes)
    planned = sum(end - start for start, end, _ in runs)
    if verbose:
        print(
            f"extracting {len(members):,} members in {len(runs):,} range request(s); "
            f"~{planned / 1e9:.3f} GB to transfer",
            flush=True,
        )

    written = 0
    bytes_out = 0
    extra_requests = 0
    renamed: list[tuple[str, str]] = []
    t0 = time.time()

    for idx, (start, end, group) in enumerate(runs, 1):
        buf = http_get(url, start, end - 1)
        for m in group:
            data, extra = _member_data(buf, m, start, url)
            extra_requests += extra
            rel = (
                m.name[len(strip_prefix) :]
                if m.name.startswith(strip_prefix)
                else m.name
            )
            if fix_image_extensions:
                stem, ext = os.path.splitext(rel)
                real = sniff_image_ext(data[:8])
                if (
                    real
                    and ext.lower() in {".jpg", ".jpeg", ".png"}
                    and real != ext.lower()
                ):
                    renamed.append((rel, stem + real))
                    rel = stem + real
            dst = os.path.join(out_dir, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as fh:
                fh.write(data)
            written += 1
            bytes_out += len(data)
        if verbose and (idx % 25 == 0 or idx == len(runs)):
            rate = planned / max(time.time() - t0, 1e-6) / 1e6
            print(
                f"  run {idx}/{len(runs)}  files={written:,}  "
                f"out={bytes_out / 1e9:.3f} GB  ~{rate:.1f} MB/s",
                flush=True,
            )

    if renamed and verbose:
        print(
            f"  corrected {len(renamed)} file extension(s) after magic-byte sniff "
            f"(e.g. {renamed[0][0]} -> {renamed[0][1]})",
            flush=True,
        )

    return {
        "members": written,
        "bytes_written": bytes_out,
        "bytes_transferred_planned": planned,
        "runs": len(runs),
        "extra_requests": extra_requests,
        "renamed": renamed[:50],
        "renamed_total": len(renamed),
        "seconds": round(time.time() - t0, 1),
    }
