#!/usr/bin/env python3
"""Build a lossless overlay GGUF for a DeepSeek-V4.1 GGUF (e.g. unsloth's UD-Q4_K_XL).

The overlay gives the dense weights that a quantized GGUF stores as Q8_0 their checkpoint values back:
  - FP8 linears (E4M3 with one UE8M0 scale per 32x32 block): the E4M3 bytes as they are, the scales as exact F32
    (2^(e - 127)), computed by ggml_mul_mat_fp8_block (CPU, CUDA)
  - BF16 linears and the embedding / lm head: the BF16 values as they are (--no-bf16 leaves them out)
Load it next to the model: llama-server -m <first shard> --shard <overlay>.
"""

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

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ARCH = "deepseek41"
SPLIT = re.compile(r"(.*)-00001-of-(\d{5})\.gguf")

# checkpoint name (layers.{i}.<key>.weight) -> GGUF name (blk.{i}.<value>.weight); scales are <key>.scale
FP8_LAYER = {
    "attn.wq_a": "attn_q_a",
    "attn.wq_b": "attn_q_b",
    "attn.wkv": "attn_kv",
    "attn.wo_a": "attn_output_a",
    "attn.wo_b": "attn_output_b",
    "attn.indexer.wq_b": "indexer.attn_q_b",
    "engram.wkv": "engram_kv",
    "ffn.shared_experts.w1": "ffn_gate_shexp",
    "ffn.shared_experts.w2": "ffn_down_shexp",
    "ffn.shared_experts.w3": "ffn_up_shexp",
}
BF16_LAYER = {
    "attn.compressor.wkv": "attn_compressor_kv",
    "attn.compressor.wgate": "attn_compressor_gate",
    "attn.indexer.wk": "indexer.attn_k",
    "attn.indexer.weights_proj": "indexer.proj",
}
BF16_ROOT = {
    "embed.weight": "token_embd.weight",
    "head.weight": "output.weight",
}
LAYER = re.compile(r"layers\.(\d+)\.(.+)\.weight")


class Checkpoint:
    def __init__(self, source: Path | None, repo: str, revision: str):
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
            info = self.session.get(f"https://huggingface.co/api/models/{repo}/revision/{revision}", timeout=60)
            info.raise_for_status()
            self.base_url = f"https://huggingface.co/{repo}/resolve/{info.json()['sha']}"

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
                }, stream=True, timeout=300,
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
    shapes = {t.name: tuple(int(d) for d in t.shape) for t in reader.tensors}
    for path in paths[1:]:
        if not path.exists():
            continue  # e.g. the Engram table shard, given separately with --ple
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name in shapes:
                raise ValueError(f"duplicate base GGUF tensor: {tensor.name}")
            shapes[tensor.name] = tuple(int(d) for d in tensor.shape)
    return shapes


def select(config: dict, index: dict, base: dict, bf16: bool, include: re.Pattern | None) -> list[tuple[str, str, str]]:
    """Returns (GGUF name, checkpoint weight name, kind) with kind 'fp8' or 'bf16', in checkpoint file order."""
    quant = config.get("quantization_config") or {}
    if (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]
            or quant.get("scale_fmt") != "ue8m0"):
        raise ValueError("expected an FP8 checkpoint with 32x32 UE8M0 weight scales")
    picked = []
    for weight in index:
        kind = target = None
        if weight in BF16_ROOT:
            kind, target = "bf16", BF16_ROOT[weight]
        elif (m := LAYER.fullmatch(weight)) is not None:
            if m[2] in FP8_LAYER:
                kind, target = "fp8", f"blk.{m[1]}.{FP8_LAYER[m[2]]}.weight"
            elif m[2] in BF16_LAYER:
                kind, target = "bf16", f"blk.{m[1]}.{BF16_LAYER[m[2]]}.weight"
        if kind is None or (kind == "bf16" and not bf16) or (include and not include.search(target)):
            continue
        if target not in base:
            raise ValueError(f"the base GGUF has no {target} for {weight}")
        if kind == "fp8" and weight.removesuffix(".weight") + ".scale" not in index:
            raise ValueError(f"missing scale for {weight}")
        picked.append((target, weight, kind))
    if not any(kind == "fp8" for _, _, kind in picked):
        raise ValueError("no FP8 source weights found")
    return sorted(picked, key=lambda p: (index[p[1]], p[1]))


def e8m0_to_f32(raw: bytes, name: str) -> np.ndarray:
    e = np.frombuffer(raw, dtype=np.uint8)
    # 2^(e - 127) is exact in F32 for every e (e = 0 is a subnormal); e4m3 * 2^(e - 127) stays finite below 246
    if e.size and int(e.max()) >= 246:
        raise ValueError(f"{name}: E8M0 scale exponent {int(e.max())} would overflow F32 or is NaN")
    return np.ldexp(np.float32(1.0), e.astype(np.int32) - 127).astype("<f4")


def write_overlay(checkpoint: Checkpoint, first: Path, output: Path, bf16: bool, include: re.Pattern | None) -> None:
    if output.exists():
        raise FileExistsError(output)
    base = gguf_shapes(first)
    config = checkpoint.read_json("config.json")
    index = checkpoint.read_json("model.safetensors.index.json")["weight_map"]
    picked = select(config, index, base, bf16, include)
    headers = {}

    def locate(source: str, st_dtype: str, shape: tuple[int, ...], itemsize: int) -> tuple[str, int, int]:
        shard = index[source]
        if shard not in headers:
            headers[shard] = checkpoint.tensor_header(shard)
        header, start = headers[shard]
        meta = header[source]
        begin, end = meta["data_offsets"]
        size = math.prod(shape) * itemsize
        if meta["dtype"] != st_dtype or tuple(meta["shape"]) != shape or begin < 0 or end - begin != size:
            raise ValueError(f"invalid source tensor layout for {source}: {meta['dtype']} {meta['shape']}")
        return shard, start + begin, size

    writer = gguf.GGUFWriter(None, ARCH)
    writer.add_bool(f"{ARCH}.fp8_block_overlay", True)
    jobs = []
    n_fp8 = n_bf16 = 0
    for target, weight, kind in picked:
        k, n = base[target]  # GGUF order: [K, N]; the checkpoint stores [N, K]
        if kind == "bf16":
            shard, offset, size = locate(weight, "BF16", (n, k), 2)
            writer.add_tensor_info(target, (n, k), np.dtype("<u2"), size, raw_dtype=gguf.GGMLQuantizationType.BF16)
            jobs.append((target, shard, offset, size, np.dtype("<u2"), (n, k), None))
            n_bf16 += 1
            continue
        if k % 32 or n % 32:
            raise ValueError(f"invalid base matrix shape for {target}: {(k, n)}")
        shard, offset, size = locate(weight, "F8_E4M3", (n, k), 1)
        writer.add_tensor_info(target, (n, k), np.dtype("u1"), size, raw_dtype=gguf.GGMLQuantizationType.F8_E4M3)
        jobs.append((target, shard, offset, size, np.dtype("u1"), (n, k), None))
        scale = weight.removesuffix(".weight") + ".scale"
        sshape = (n // 32, k // 32)
        shard, offset, size = locate(scale, "F8_E8M0", sshape, 1)
        sname = target.removesuffix(".weight") + ".fp8_block_scale"
        writer.add_tensor_info(sname, sshape, np.dtype("<f4"), size * 4, raw_dtype=gguf.GGMLQuantizationType.F32)
        jobs.append((sname, shard, offset, size, np.dtype("<f4"), sshape, scale))
        n_fp8 += 1

    print(f"Writing {n_fp8} FP8 weight/scale pairs and {n_bf16} BF16 weights to {output}", flush=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".gguf", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        writer.write_header_to_file(temp_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for name, shard, offset, size, dtype, shape, e8m0 in jobs:
            data = checkpoint.read_range(shard, offset, size)
            array = e8m0_to_f32(data, e8m0) if e8m0 else np.frombuffer(data, dtype=dtype)
            writer.write_tensor_data(array.reshape(shape), gguf.GGUFEndian.LITTLE)
        writer.close()
        umask = os.umask(0); os.umask(umask)
        os.chmod(temp_path, 0o666 & ~umask)
        os.link(temp_path, output)
    finally:
        writer.close()
        temp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="first shard of the base GGUF")
    parser.add_argument("output", type=Path, help="output overlay GGUF")
    parser.add_argument("--source-dir", type=Path, help="local DeepSeek safetensors directory")
    parser.add_argument("--repo", default=REPO, help="Hugging Face repo of the remote source")
    parser.add_argument("--revision", default="main", help="Hugging Face revision for the remote source")
    parser.add_argument("--no-bf16", action="store_true", help="only the FP8 weights, keep the base's BF16 ones")
    parser.add_argument("--include", type=re.compile, help="only the GGUF tensors whose name matches this regex")
    args = parser.parse_args()
    write_overlay(Checkpoint(args.source_dir, args.repo, args.revision), args.base, args.output,
                  not args.no_bf16, args.include)


if __name__ == "__main__":
    main()
