#!/usr/bin/env python3
"""Run bounded production workflow checks against an ephemeral HTTP server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import struct
import sys
import tempfile
import threading
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avprovenance.http_api import ProvenanceHTTPServer, build_http_server


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_wav(
    *,
    frequency: float,
    duration_seconds: float = 0.6,
    sample_rate: int = 8000,
    amplitude: int = 9000,
) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    values = [
        int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
        for index in range(frame_count)
    ]
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(values)}h", *values))
    return payload.getvalue()


def make_y4m(
    *,
    width: int = 64,
    height: int = 36,
    frame_count: int = 5,
    rate_numerator: int = 12,
    rate_denominator: int = 1,
    pixel_format: str = "C420jpeg",
) -> bytes:
    header = (
        "YUV4MPEG2 "
        f"W{width} H{height} "
        f"F{rate_numerator}:{rate_denominator} "
        f"Ip {pixel_format}\n"
    ).encode("ascii")
    ten_bit = pixel_format == "C420p10"
    bytes_per_sample = 2 if ten_bit else 1
    sample_limit = 1024 if ten_bit else 256

    def encode_sample(value: int) -> bytes:
        return (value % sample_limit).to_bytes(bytes_per_sample, "little")

    chroma_width = width // 2
    chroma_height = height // 2
    parts = [header]
    for frame_index in range(frame_count):
        parts.append(b"FRAME\n")
        y_plane = b"".join(
            encode_sample(x * 3 + y * 5 + frame_index * 11)
            for y in range(height)
            for x in range(width)
        )
        u_plane = b"".join(
            encode_sample(96 + x * 2 + frame_index * 3)
            for _y in range(chroma_height)
            for x in range(chroma_width)
        )
        v_plane = b"".join(
            encode_sample(144 + y * 2 + frame_index * 5)
            for y in range(chroma_height)
            for _x in range(chroma_width)
        )
        parts.extend((y_plane, u_plane, v_plane))
    return b"".join(parts)


class HTTPClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=15,
        )
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            return response.status, value
        finally:
            connection.close()

    def upload(self, filename: str, data: bytes) -> dict[str, Any]:
        status, body = self.request(
            "POST",
            "/v1/assets",
            {
                "filename": filename,
                "data_base64": base64.b64encode(data).decode("ascii"),
            },
        )
        require_equal(status, 201, "asset upload status")
        return body

    def create_rendition(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body = self.request("POST", "/v1/renditions", payload)
        require_equal(status, 201, "rendition creation status")
        return body


@contextmanager
def running_server() -> Iterator[tuple[HTTPClient, ProvenanceHTTPServer]]:
    with tempfile.TemporaryDirectory(prefix="avprovenance-check-") as temporary:
        server = build_http_server("127.0.0.1", 0, Path(temporary))
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        thread.start()
        host, port = server.server_address[:2]
        try:
            yield HTTPClient(str(host), int(port)), server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_ingest_media_asset(client: HTTPClient) -> dict[str, Any]:
    health_status, health = client.request("GET", "/health")
    require_equal(health_status, 200, "health status")
    require_equal(health["status"], "ok", "health payload")

    audio = make_wav(frequency=330)
    asset = client.upload("tone-source.wav", audio)
    require_equal(asset["kind"], "audio", "asset kind")
    require_equal(asset["sha256"], _sha256(audio), "asset digest")
    require_equal(asset["descriptor"]["container"], "wav", "asset container")
    require("content_key" not in asset, "public asset exposed an internal content key")

    get_status, fetched = client.request("GET", f"/v1/assets/{asset['id']}")
    require_equal(get_status, 200, "asset get status")
    require_equal(fetched["sha256"], asset["sha256"], "asset read digest")

    bad_status, bad_body = client.request(
        "POST",
        "/v1/assets",
        {
            "filename": "unknown.bin",
            "data_base64": base64.b64encode(b"not-media").decode("ascii"),
        },
    )
    require_equal(bad_status, 415, "unsupported media status")
    require_equal(
        bad_body["error"]["code"],
        "UNSUPPORTED_MEDIA",
        "unsupported media code",
    )
    return {
        "workflow": "ingest-media-asset",
        "asset_id": asset["id"],
        "kind": asset["kind"],
        "digest": asset["sha256"],
    }


def run_execute_rendition(client: HTTPClient) -> dict[str, Any]:
    primary_bytes = make_wav(frequency=440, amplitude=10000)
    secondary_bytes = make_wav(frequency=220, amplitude=7000)
    primary = client.upload("primary-tone.wav", primary_bytes)
    secondary = client.upload("secondary-tone.wav", secondary_bytes)

    audio_rendition = client.create_rendition(
        {
            "primary_asset_id": primary["id"],
            "secondary_asset_id": secondary["id"],
            "output_name": "mixed-tone.wav",
            "operations": [
                {
                    "type": "media_trim",
                    "start_seconds": 0.05,
                    "duration_seconds": 0.3,
                },
                {"type": "audio_gain", "db": -3},
                {
                    "type": "audio_mix",
                    "secondary_gain": -6,
                    "primary_weight": 0.75,
                    "secondary_weight": 0.4,
                },
                {"type": "audio_normalize", "target_peak": 0.8},
            ],
        }
    )
    require_equal(audio_rendition["status"], "completed", "audio rendition status")
    history = [item["to"] for item in audio_rendition["transitions"]]
    require_equal(
        history,
        ["queued", "running", "completed"],
        "audio state history",
    )
    require(
        audio_rendition["output_asset"]["descriptor"]["container"] == "wav",
        "audio output descriptor was not WAV",
    )

    video_bytes = make_y4m()
    video = client.upload("source-frames.y4m", video_bytes)
    video_rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "cropped-frames.y4m",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 8,
                    "y": 6,
                    "width": 32,
                    "height": 18,
                },
                {"type": "video_luma_offset", "amount": 12},
                {"type": "video_mirror"},
            ],
        }
    )
    require_equal(video_rendition["status"], "completed", "video rendition status")
    properties = video_rendition["output_asset"]["descriptor"]["properties"]
    require_equal(properties["width"], 32, "video output width")
    require_equal(properties["height"], 18, "video output height")

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{audio_rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "lineage status")
    require_equal(len(lineage["parents"]), 2, "lineage parent count")
    require_equal(
        lineage["edge"]["plan_digest"],
        audio_rendition["plan_digest"],
        "lineage plan digest",
    )
    invalid_status, invalid_body = client.request(
        "POST",
        "/v1/renditions",
        {
            "primary_asset_id": primary["id"],
            "output_name": "invalid.wav",
            "operations": [
                {
                    "type": "video_crop",
                    "x": 0,
                    "y": 0,
                    "width": 2,
                    "height": 2,
                }
            ],
        },
    )
    require_equal(invalid_status, 422, "invalid plan status")
    require_equal(invalid_body["error"]["code"], "PLAN_INVALID", "invalid plan code")
    return {
        "workflow": "execute-rendition",
        "audio_rendition_id": audio_rendition["id"],
        "video_rendition_id": video_rendition["id"],
        "audio_history": history,
        "invalid_plan_code": invalid_body["error"]["code"],
        "video_dimensions": [
            properties["width"],
            properties["height"],
        ],
    }


def run_verify_reproduction(
    client: HTTPClient,
    server: ProvenanceHTTPServer,
) -> dict[str, Any]:
    video_bytes = make_y4m(frame_count=7)
    video = client.upload("verification-source.y4m", video_bytes)
    rendition = client.create_rendition(
        {
            "primary_asset_id": video["id"],
            "output_name": "verification-output.y4m",
            "operations": [
                {"type": "media_trim", "start_frame": 1, "frame_count": 4},
                {
                    "type": "video_crop",
                    "x": 4,
                    "y": 4,
                    "width": 40,
                    "height": 20,
                },
                {"type": "video_luma_offset", "amount": -9},
                {"type": "video_mirror"},
            ],
        }
    )
    check_status, check = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(check_status, 201, "integrity check status")
    require_equal(check["status"], "verified", "integrity result")
    require_equal(
        check["expected_digest"],
        check["reproduced_digest"],
        "integrity digest equality",
    )
    require_equal(
        check["expected_descriptor"],
        check["reproduced_descriptor"],
        "integrity descriptor equality",
    )

    lineage_status, lineage = client.request(
        "GET",
        f"/v1/renditions/{rendition['id']}/lineage",
    )
    require_equal(lineage_status, 200, "verification lineage status")
    require_equal(len(lineage["parents"]), 1, "verification parent count")

    output_asset_id = rendition["output_asset"]["id"]

    def drift_metadata(state: dict[str, Any]) -> None:
        state["assets"][output_asset_id]["sha256"] = "0" * 64

    server.application.state_file.mutate(drift_metadata)
    mismatch_status, mismatch = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(mismatch_status, 409, "mismatch status")
    require_equal(
        mismatch["error"]["code"],
        "INTEGRITY_MISMATCH",
        "mismatch error code",
    )
    require_equal(
        mismatch["error"]["details"]["audit"]["status"],
        "mismatch",
        "persisted mismatch status",
    )
    return {
        "workflow": "verify-reproduction",
        "rendition_id": rendition["id"],
        "integrity_check_id": check["id"],
        "digest": check["reproduced_digest"],
        "mismatch_status": mismatch["error"]["details"]["audit"]["status"],
    }


def _y4m_planes(data: bytes) -> dict[str, Any]:
    """Decode a single-family Y4M payload into geometry and integer samples."""
    from avprovenance.media.y4m import decode_y4m

    clip = decode_y4m(data)
    bytes_per_sample = clip.bytes_per_sample

    def samples(plane: bytes) -> list[int]:
        if bytes_per_sample == 1:
            return list(plane)
        return [
            plane[index] | (plane[index + 1] << 8)
            for index in range(0, len(plane), 2)
        ]

    frame = clip.frames[0]
    return {
        "clip": clip,
        "width": clip.width,
        "height": clip.height,
        "chroma_width": clip.chroma_width,
        "chroma_height": clip.chroma_height,
        "bit_depth": clip.bit_depth,
        "y": samples(frame.y_plane),
        "u": samples(frame.u_plane),
        "v": samples(frame.v_plane),
    }


def run_ten_bit_video_pipeline(
    client: HTTPClient,
    server: ProvenanceHTTPServer,
) -> dict[str, Any]:
    # ---- 10-bit ingest is proven by descriptor and by stored file content ----
    source_bytes = make_y4m(
        width=64,
        height=36,
        frame_count=6,
        pixel_format="C420p10",
    )
    source = client.upload("ten-bit-source.y4m", source_bytes)
    source_properties = source["descriptor"]["properties"]
    require_equal(
        source_properties["pixel_format"],
        "C420p10",
        "10-bit source pixel format",
    )
    require_equal(source_properties["bit_depth"], 10, "10-bit source bit depth")
    stored_source = server.application._asset_bytes(
        server.application.repository.get_asset(source["id"])
    )
    require_equal(stored_source, source_bytes, "10-bit source bytes preserved")

    # ---- full crop / luma / mirror chain stays in a true 10-bit container ----
    operations = [
        {"type": "media_trim", "start_frame": 1, "frame_count": 4},
        {
            "type": "video_crop",
            "x": 4,
            "y": 4,
            "width": 40,
            "height": 20,
        },
        {"type": "video_luma_offset", "amount": 12},
        {"type": "video_mirror"},
    ]
    rendition = client.create_rendition(
        {
            "primary_asset_id": source["id"],
            "output_name": "ten-bit-derived.y4m",
            "operations": operations,
        }
    )
    require_equal(rendition["status"], "completed", "10-bit rendition status")
    output_asset = rendition["output_asset"]
    output_properties = output_asset["descriptor"]["properties"]
    require_equal(
        output_properties["pixel_format"],
        "C420p10",
        "10-bit output pixel format",
    )
    require_equal(output_properties["bit_depth"], 10, "10-bit output bit depth")
    require_equal(output_properties["width"], 40, "10-bit output width")
    require_equal(output_properties["height"], 20, "10-bit output height")

    output_bytes = server.application._asset_bytes(
        server.application.repository.get_asset(output_asset["id"])
    )
    # The on-disk derivative is genuinely 16-bit-per-sample C420p10, not an
    # 8-bit stream wearing a new label.
    parsed = _y4m_planes(output_bytes)
    require_equal(parsed["bit_depth"], 10, "decoded output bit depth")
    require(
        all(0 <= sample <= 1023 for sample in parsed["y"]),
        "10-bit luma samples stay inside the 10-bit range",
    )
    require(
        all(0 <= sample <= 1023 for sample in parsed["u"] + parsed["v"]),
        "10-bit chroma samples stay inside the 10-bit range",
    )
    require(
        any(sample > 255 for sample in parsed["y"] + parsed["u"] + parsed["v"]),
        "10-bit output carries values that an 8-bit container could not hold",
    )

    # ---- luma magnitude matches the 8-bit scale (x4) and never wraps ----
    source_parsed = _y4m_planes(source_bytes)
    luma_only = client.create_rendition(
        {
            "primary_asset_id": source["id"],
            "output_name": "ten-bit-luma.y4m",
            "operations": [{"type": "video_luma_offset", "amount": 12}],
        }
    )
    luma_bytes = server.application._asset_bytes(
        server.application.repository.get_asset(luma_only["output_asset"]["id"])
    )
    luma_parsed = _y4m_planes(luma_bytes)
    for original, adjusted in zip(source_parsed["y"], luma_parsed["y"]):
        require_equal(
            adjusted,
            max(0, min(1023, original + 48)),
            "10-bit luma offset is scaled x4 and clamped without wrap",
        )

    # ---- mirror reverses samples, never the bytes inside a 16-bit slot ----
    mirror_only = client.create_rendition(
        {
            "primary_asset_id": source["id"],
            "output_name": "ten-bit-mirror.y4m",
            "operations": [{"type": "video_mirror"}],
        }
    )
    mirror_bytes = server.application._asset_bytes(
        server.application.repository.get_asset(mirror_only["output_asset"]["id"])
    )
    mirror_parsed = _y4m_planes(mirror_bytes)
    width = mirror_parsed["width"]
    source_y = source_parsed["y"]
    mirrored_y = mirror_parsed["y"]
    for row in range(mirror_parsed["height"]):
        source_row = source_y[row * width : (row + 1) * width]
        mirrored_row = mirrored_y[row * width : (row + 1) * width]
        require_equal(
            mirrored_row,
            source_row[::-1],
            "10-bit horizontal mirror reverses whole samples",
        )

    # ---- integrity replay proves bit depth through bytes and descriptor ----
    check_status, check = client.request(
        "POST",
        f"/v1/renditions/{rendition['id']}/integrity-checks",
        {},
    )
    require_equal(check_status, 201, "10-bit integrity check status")
    require_equal(check["status"], "verified", "10-bit integrity result")
    require_equal(
        check["reproduced_digest"],
        output_asset["sha256"],
        "10-bit replayed digest",
    )
    require_equal(
        check["reproduced_descriptor"],
        output_asset["descriptor"],
        "10-bit replayed descriptor",
    )

    # ---- the generated derivative is itself a valid source that stays 10-bit ----
    derived_asset = client.upload("ten-bit-derived-reingest.y4m", output_bytes)
    require_equal(
        derived_asset["descriptor"]["properties"]["bit_depth"],
        10,
        "re-ingested derivative bit depth",
    )
    second_rendition = client.create_rendition(
        {
            "primary_asset_id": derived_asset["id"],
            "output_name": "ten-bit-derived-again.y4m",
            "operations": [
                {"type": "video_mirror"},
                {"type": "video_luma_offset", "amount": -4},
            ],
        }
    )
    require_equal(
        second_rendition["output_asset"]["descriptor"]["properties"]["bit_depth"],
        10,
        "second-generation 10-bit bit depth",
    )
    second_check_status, second_check = client.request(
        "POST",
        f"/v1/renditions/{second_rendition['id']}/integrity-checks",
        {},
    )
    require_equal(
        second_check_status,
        201,
        "second-generation integrity check status",
    )
    require_equal(
        second_check["status"],
        "verified",
        "second-generation integrity result",
    )

    # ---- malformed 10-bit payloads (sample outside 10 bits) are rejected ----
    malformed = bytearray(source_bytes)
    # Flip a high container bit in the first luma sample's high byte.
    frame_pos = malformed.index(b"FRAME\n") + len(b"FRAME\n")
    malformed[frame_pos + 1] |= 0x04
    bad_status, bad_body = client.request(
        "POST",
        "/v1/assets",
        {
            "filename": "bad-ten-bit.y4m",
            "data_base64": base64.b64encode(bytes(malformed)).decode("ascii"),
        },
    )
    require_equal(bad_status, 415, "malformed 10-bit ingest status")
    require_equal(
        bad_body["error"]["code"],
        "UNSUPPORTED_MEDIA",
        "malformed 10-bit ingest code",
    )

    return {
        "workflow": "ten-bit-video-pipeline",
        "source_asset_id": source["id"],
        "rendition_id": rendition["id"],
        "output_digest": output_asset["sha256"],
        "bit_depth": output_properties["bit_depth"],
        "integrity": check["status"],
        "second_generation_integrity": second_check["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workflow",
        choices=(
            "ingest-media-asset",
            "execute-rendition",
            "verify-reproduction",
            "ten-bit-video-pipeline",
        ),
    )
    args = parser.parse_args()

    with running_server() as (client, server):
        if args.workflow == "ingest-media-asset":
            result = run_ingest_media_asset(client)
        elif args.workflow == "execute-rendition":
            result = run_execute_rendition(client)
        elif args.workflow == "ten-bit-video-pipeline":
            result = run_ten_bit_video_pipeline(client, server)
        else:
            result = run_verify_reproduction(client, server)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
