#!/usr/bin/env python3
"""Build a block-scaled FP8 overlay for a GLM-5.3-Flash GGUF."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parent / "gguf-py"))
import gguf

REPO = "zai-org/GLM-5.3-Flash"
ARCH = "glm5-next"
BLOCK = 128
SCALE_SUFFIX = "_scale_inv"
SPLIT = re.compile(r"(.*)-00001-of-(\d{5})\.gguf")
TARGET = re.compile(
    r"blk\.\d+\.(attn_(kv_a_mqa|output|q_a|q_b)|ffn_(gate|up|down)(_shexp)?)\.weight"
)


class Checkpoint:
    def __init__(self, source: Path | None, revision: str):
        self.source = source
        if source is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            self.session = requests.Session()
            self.session.mount("https://", HTTPAdapter(max_retries=Retry(
                total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)
            )))
            token = os.environ.get("HF_TOKEN")
            if token:
                self.session.headers["Authorization"] = f"Bearer {token}"
            info = self.session.get(f"https://huggingface.co/api/models/{REPO}/revision/{revision}", timeout=60)
            info.raise_for_status()
            self.base_url = f"https://huggingface.co/{REPO}/resolve/{info.json()['sha']}"

    def read_json(self, name: str) -> dict:
        if self.source is not None:
            return json.loads((self.source / name).read_bytes())
        response = self.session.get(f"{self.base_url}/{name}", timeout=60)
        response.raise_for_status()
        return response.json()

    def read_range(self, name: str, offset: int, size: int) -> bytes:
        if self.source is not None:
            with (self.source / name).open("rb") as file:
                file.seek(offset)
                data = file.read(size)
        else:
            end = offset + size - 1
            with self.session.get(
                f"{self.base_url}/{name}", headers={
                    "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
                }, stream=True, timeout=120,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-{end}/"
                ):
                    raise ValueError(f"server did not honor range {offset}-{end} for {name}")
                data = response.content
        if len(data) != size:
            raise ValueError(f"short read in {name} at {offset}")
        return data

    def tensor_header(self, name: str) -> tuple[dict, int]:
        size = int.from_bytes(self.read_range(name, 0, 8), "little")
        if not 0 < size <= 64 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length in {name}: {size}")
        return json.loads(self.read_range(name, 8, size)), size + 8


def gguf_shapes(first: Path) -> dict[str, tuple[int, ...]]:
    reader = gguf.GGUFReader(first)
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != ARCH:
        raise ValueError(f"{first} must be a {ARCH} GGUF")
    split_count = reader.get_field("split.count")
    count = split_count.contents() if split_count else 1
    paths = [first]
    if count > 1:
        match = SPLIT.fullmatch(first.name)
        if match is None or int(match[2]) != count:
            raise ValueError("provide the first numbered GGUF shard")
        paths.extend(first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(2, count + 1))
    shapes = {t.name: tuple(t.shape) for t in reader.tensors}
    for path in paths[1:]:
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name in shapes:
                raise ValueError(f"duplicate base GGUF tensor: {tensor.name}")
            shapes[tensor.name] = tuple(tensor.shape)
    return shapes


def select_pairs(config: dict, index: dict, base: dict) -> list[tuple[str, str]]:
    quant = config["quantization_config"]
    if quant.get("fmt") != "e4m3" or quant.get("weight_block_size") != [BLOCK, BLOCK]:
        raise ValueError("expected E4M3 with 128x128 weight scales")
    text = config["text_config"]
    trunk_layers = text["num_hidden_layers"]
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GLM5_NEXT, trunk_layers + text["num_nextn_predict_layers"])
    pairs = []
    for scale in index:
        if not scale.endswith(".weight" + SCALE_SUFFIX) or ".mlp.experts." in scale:
            continue
        weight = scale.removesuffix(SCALE_SUFFIX)
        hf_name = weight.replace("model.language_model.", "model.", 1)
        name = tensor_map.get_name(hf_name, try_suffixes=(".weight",))
        if name is None or TARGET.fullmatch(name) is None:
            raise ValueError(f"unsupported FP8 source tensor: {weight}")
        layer = int(name.split(".")[1])
        if name not in base and layer >= trunk_layers:
            continue  # The base may have been exported with --no-mtp.
        if name not in base or weight not in index:
            raise ValueError(f"missing base or source tensor for {weight}")
        pairs.append((name, weight))
    if not pairs:
        raise ValueError("no eligible source FP8 weights")
    return sorted(pairs, key=lambda pair: (index[pair[1]], pair[0]))


def write_overlay(checkpoint: Checkpoint, first: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    base = gguf_shapes(first)
    config = checkpoint.read_json("config.json")
    index = checkpoint.read_json("model.safetensors.index.json")["weight_map"]
    pairs = select_pairs(config, index, base)
    headers = {}
    writer = gguf.GGUFWriter(None, ARCH)
    writer.add_bool(f"{ARCH}.fp8_block_overlay", True)
    ranges = []
    for name, weight in pairs:
        k, n = base[name]
        if k % BLOCK or n % BLOCK:
            raise ValueError(f"invalid base matrix shape for {name}: {(k, n)}")
        specs = (
            (name, weight, "F8_E4M3", (n, k), np.dtype("u1"), gguf.GGMLQuantizationType.F8_E4M3),
            (name.removesuffix(".weight") + ".fp8_block_scale", weight + SCALE_SUFFIX,
             "F32", (n // BLOCK, k // BLOCK), np.dtype("<f4"), gguf.GGMLQuantizationType.F32),
        )
        for target, source, st_dtype, shape, dtype, ggml_type in specs:
            shard = index[source]
            if shard not in headers:
                headers[shard] = checkpoint.tensor_header(shard)
            header, start = headers[shard]
            meta = header[source]
            begin, end = meta["data_offsets"]
            size = math.prod(shape) * dtype.itemsize
            if meta["dtype"] != st_dtype or tuple(meta["shape"]) != shape or begin < 0 or end - begin != size:
                raise ValueError(f"invalid source tensor layout for {source}")
            writer.add_tensor_info(target, shape, dtype, size, raw_dtype=ggml_type)
            ranges.append((shard, start + begin, size, dtype, shape))

    print(f"Writing {len(pairs)} weight/scale pairs to {output}", flush=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".gguf", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        writer.write_header_to_file(temp_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for shard, offset, size, dtype, shape in ranges:
            data = checkpoint.read_range(shard, offset, size)
            writer.write_tensor_data(np.frombuffer(data, dtype=dtype).reshape(shape), gguf.GGUFEndian.LITTLE)
        writer.close()
        os.link(temp_path, output)
    finally:
        writer.close()
        temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="first shard of the base GGUF")
    parser.add_argument("output", type=Path, help="output overlay GGUF")
    parser.add_argument("--source-dir", type=Path, help="local Z.ai safetensors directory")
    parser.add_argument("--revision", default="main", help="Hugging Face revision for the remote source")
    args = parser.parse_args()
    write_overlay(Checkpoint(args.source_dir, args.revision), args.base, args.output)


if __name__ == "__main__":
    main()
