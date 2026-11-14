"""Planar Y4M operations that preserve deterministic frame layout.

Every operation is bit-depth aware. For 8-bit C420 inputs the byte-level
behaviour is unchanged. C420p10 inputs keep their little-endian 16-bit
sample containers: geometry is computed in samples, mirroring reverses
samples rather than bytes, and luma offsets are scaled to the same visual
amplitude and clamped instead of allowed to wrap.
"""

from __future__ import annotations

from typing import Any

from .errors import MediaOperationFailure
from .y4m import Y4mClip, Y4mFrame, decode_y4m, encode_y4m


# Reference luma scale used by the historical 8-bit operation. A requested
# offset is expressed on this scale so 10-bit renders move the same visual
# amount (one 8-bit code unit == four 10-bit code units).
EIGHT_BIT_LUMA_SCALE = 8


def trim_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    start = int(operation["start_frame"])
    count = int(operation["frame_count"])
    if start < 0 or start >= clip.frame_count:
        raise MediaOperationFailure(
            "VIDEO_TRIM_OUT_OF_RANGE",
            "The requested video trim starts outside the source.",
        )
    end = min(clip.frame_count, start + max(1, count))
    frames = clip.frames[start:end]
    if not frames:
        raise MediaOperationFailure(
            "VIDEO_TRIM_EMPTY",
            "The requested video trim produces no frames.",
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
        bit_depth=clip.bit_depth,
    )


def _crop_plane(
    plane: bytes,
    plane_width: int,
    x: int,
    y: int,
    width: int,
    height: int,
    bytes_per_sample: int,
) -> bytes:
    row_stride = plane_width * bytes_per_sample
    x_offset = x * bytes_per_sample
    row_width = width * bytes_per_sample
    rows = []
    for row_index in range(y, y + height):
        start = row_index * row_stride + x_offset
        rows.append(plane[start : start + row_width])
    return b"".join(rows)


def crop_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    x = int(operation["x"])
    y = int(operation["y"])
    width = int(operation["width"])
    height = int(operation["height"])
    if any(value % 2 for value in (x, y, width, height)):
        raise MediaOperationFailure(
            "VIDEO_CROP_ALIGNMENT",
            "Video crop geometry must align to even pixels.",
        )
    if width < 2 or height < 2:
        raise MediaOperationFailure(
            "VIDEO_CROP_TOO_SMALL",
            "Crop width and height must be at least two pixels.",
        )
    if x + width > clip.width or y + height > clip.height:
        raise MediaOperationFailure(
            "VIDEO_CROP_OUT_OF_RANGE",
            "Crop geometry exceeds the source frame.",
        )

    chroma_x = x // 2
    chroma_y = y // 2
    chroma_width = width // 2
    chroma_height = height // 2
    bytes_per_sample = clip.bytes_per_sample
    frames = []
    for frame in clip.frames:
        frames.append(
            Y4mFrame(
                y_plane=_crop_plane(
                    frame.y_plane,
                    clip.width,
                    x,
                    y,
                    width,
                    height,
                    bytes_per_sample,
                ),
                u_plane=_crop_plane(
                    frame.u_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                    bytes_per_sample,
                ),
                v_plane=_crop_plane(
                    frame.v_plane,
                    clip.chroma_width,
                    chroma_x,
                    chroma_y,
                    chroma_width,
                    chroma_height,
                    bytes_per_sample,
                ),
            )
        )
    return Y4mClip(
        width=width,
        height=height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
        bit_depth=clip.bit_depth,
    )


def _offset_luma_eight_bit(plane: bytes, amount: int) -> bytes:
    return bytes(max(0, min(255, value + amount)) for value in plane)


def _offset_luma_ten_bit(plane: bytes, amount: int) -> bytes:
    # The amount is authored on the 8-bit luma scale; rescale it to the
    # 10-bit container so the visual magnitude matches the 8-bit operation.
    scaled_amount = amount << (10 - EIGHT_BIT_LUMA_SCALE)
    adjusted = bytearray(len(plane))
    for index in range(0, len(plane), 2):
        value = plane[index] | (plane[index + 1] << 8)
        value += scaled_amount
        if value < 0:
            value = 0
        elif value > 1023:
            value = 1023
        adjusted[index] = value & 0xFF
        adjusted[index + 1] = (value >> 8) & 0xFF
    return bytes(adjusted)


def offset_luma(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    amount = int(operation["amount"])
    if clip.bit_depth == 10:
        def adjust(plane: bytes) -> bytes:
            return _offset_luma_ten_bit(plane, amount)
    else:
        def adjust(plane: bytes) -> bytes:
            return _offset_luma_eight_bit(plane, amount)
    frames = []
    for frame in clip.frames:
        frames.append(
            Y4mFrame(
                y_plane=adjust(frame.y_plane),
                u_plane=frame.u_plane,
                v_plane=frame.v_plane,
            )
        )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=tuple(frames),
        bit_depth=clip.bit_depth,
    )


def _mirror_plane(plane: bytes, width: int, height: int, bytes_per_sample: int) -> bytes:
    row_stride = width * bytes_per_sample
    if bytes_per_sample == 1:
        rows = []
        for row_index in range(height):
            start = row_index * row_stride
            rows.append(plane[start : start + row_stride][::-1])
        return b"".join(rows)

    # Reversing the raw bytes of a 16-bit little-endian plane would swap each
    # sample's low and high bytes and scatter chroma sample locations. Reverse
    # at sample granularity instead, preserving each two-byte container.
    rows = []
    for row_index in range(height):
        start = row_index * row_stride
        row = plane[start : start + row_stride]
        reversed_row = bytearray(row_stride)
        for sample_index in range(width):
            source = sample_index * 2
            target = (width - 1 - sample_index) * 2
            reversed_row[target] = row[source]
            reversed_row[target + 1] = row[source + 1]
        rows.append(bytes(reversed_row))
    return b"".join(rows)


def mirror_video(clip: Y4mClip, operation: dict[str, Any]) -> Y4mClip:
    del operation
    bytes_per_sample = clip.bytes_per_sample
    frames = tuple(
        Y4mFrame(
            y_plane=_mirror_plane(
                frame.y_plane, clip.width, clip.height, bytes_per_sample
            ),
            u_plane=_mirror_plane(
                frame.u_plane,
                clip.chroma_width,
                clip.chroma_height,
                bytes_per_sample,
            ),
            v_plane=_mirror_plane(
                frame.v_plane,
                clip.chroma_width,
                clip.chroma_height,
                bytes_per_sample,
            ),
        )
        for frame in clip.frames
    )
    return Y4mClip(
        width=clip.width,
        height=clip.height,
        rate_numerator=clip.rate_numerator,
        rate_denominator=clip.rate_denominator,
        interlace=clip.interlace,
        pixel_format=clip.pixel_format,
        frames=frames,
        bit_depth=clip.bit_depth,
    )


def render_video(
    primary_bytes: bytes,
    operations: list[dict[str, Any]],
    secondary_bytes: bytes | None,
) -> bytes:
    if secondary_bytes is not None:
        raise MediaOperationFailure(
            "VIDEO_SECONDARY_UNSUPPORTED",
            "The current video operation set accepts one source.",
        )
    clip = decode_y4m(primary_bytes)
    try:
        for operation in operations:
            operation_type = operation["type"]
            if operation_type == "media_trim":
                clip = trim_video(clip, operation)
            elif operation_type == "video_crop":
                clip = crop_video(clip, operation)
            elif operation_type == "video_luma_offset":
                clip = offset_luma(clip, operation)
            elif operation_type == "video_mirror":
                clip = mirror_video(clip, operation)
            else:
                raise MediaOperationFailure(
                    "UNSUPPORTED_VIDEO_OPERATION",
                    f"Unsupported video operation: {operation_type}",
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaOperationFailure(
            "VIDEO_OPERATION_INVALID",
            "The video operation payload is malformed.",
        ) from exc
    return encode_y4m(clip)
