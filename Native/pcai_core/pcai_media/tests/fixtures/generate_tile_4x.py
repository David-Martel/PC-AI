"""Generate the deterministic ONNX Tile fixture for the CPU upscale test."""

from argparse import ArgumentParser
from pathlib import Path
from struct import pack


def varint(value: int) -> bytes:
    octets = bytearray()
    while value >= 128:
        octets.append((value & 0x7F) | 0x80)
        value >>= 7
    octets.append(value)
    return bytes(octets)


def integer(field: int, value: int) -> bytes:
    return varint(field << 3) + varint(value)


def delimited(field: int, payload: bytes) -> bytes:
    return varint((field << 3) | 2) + varint(len(payload)) + payload


def tensor_value(name: bytes, dimensions: tuple[int, ...]) -> bytes:
    # ValueInfoProto.type.tensor_type describes a FLOAT TensorShapeProto.
    shape = b"".join(delimited(1, integer(1, size)) for size in dimensions)
    tensor_type = integer(1, 1) + delimited(2, shape)
    return delimited(1, name) + delimited(2, delimited(1, tensor_type))


def build_fixture() -> bytes:
    """Serialize ModelProto IR 8 with the Tile operator from opset 13."""
    tile = (
        delimited(1, b"rgb")
        + delimited(1, b"repeats")
        + delimited(2, b"tiled")
        + delimited(4, b"Tile")
    )
    # TensorProto has shape [4], type INT64 and little-endian raw_data.
    repeats = (
        integer(1, 4)
        + integer(2, 7)
        + delimited(8, b"repeats")
        + delimited(9, pack("<4q", 1, 1, 4, 4))
    )
    graph = (
        delimited(1, tile)
        + delimited(2, b"tile_4x")
        + delimited(5, repeats)
        + delimited(11, tensor_value(b"rgb", (1, 3, 2, 3)))
        + delimited(12, tensor_value(b"tiled", (1, 3, 8, 12)))
    )
    return (
        integer(1, 8)
        + delimited(2, b"PC-AI test fixture")
        + delimited(7, graph)
        + delimited(8, integer(2, 13))
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path to the generated tile-4x.onnx.pb")
    parser.add_argument("--check", action="store_true", help="compare existing fixture bytes without writing")
    args = parser.parse_args()
    fixture = build_fixture()
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != fixture:
            parser.error(f"fixture differs from the generator: {args.output}")
    else:
        args.output.write_bytes(fixture)
