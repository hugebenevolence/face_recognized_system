"""Which video sources the API is allowed to switch to, and how they are validated.

The /source endpoint is unauthenticated and the URI ends up inside a gst-launch
description, so it is restricted to an allow-list: RTSP URLs, or video files under
/samples. Anything else (file:///etc/passwd, http://internal-service, quote
injection into the pipeline string, ...) is rejected.
"""

import re
from pathlib import Path

SAMPLES_DIR = Path("/samples")
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
_URI_OK = re.compile(r"^[A-Za-z0-9_:/@.\-?=&%+~#,;]+$")


def validate_uri(uri: str) -> str:
    uri = (uri or "").strip()
    if not uri or len(uri) > 512 or not _URI_OK.match(uri):
        raise ValueError("Invalid source URL")
    if uri.startswith(("rtsp://", "rtsps://")):
        return uri
    if uri.startswith("/samples/") and ".." not in uri and Path(uri).suffix.lower() in VIDEO_EXT:
        if not Path(uri).is_file():
            raise ValueError("File not found")
        return uri
    raise ValueError("Use an RTSP URL or a video file from the samples folder")


def list_file_sources() -> list[dict]:
    if not SAMPLES_DIR.is_dir():
        return []
    return [
        {"id": f"file:{p.name}", "label": f"File {p.name}", "uri": str(p)}
        for p in sorted(SAMPLES_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXT
    ]


def public_uri(uri: str) -> str:
    """Never expose credentials embedded in rtsp://user:pass@host/..."""
    return uri.split("@")[-1]
