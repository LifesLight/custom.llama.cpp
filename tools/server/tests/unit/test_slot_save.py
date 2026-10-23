import pytest
from utils import *
import base64
import requests
import struct
import json
import time
from pathlib import Path

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84
    slot_file = os.path.join(server.slot_save_path, "slot1.bin")
    assert res.body["n_written"] == os.path.getsize(slot_file)

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == os.path.getsize(slot_file)

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


@pytest.mark.parametrize("save_on_shutdown", [False, True])
def test_disk_prompt_cache_restart(tmp_path, monkeypatch, save_on_shutdown):
    if save_on_shutdown and os.name == "nt":
        pytest.skip("Popen.terminate uses TerminateProcess on Windows")
    pool = tmp_path / "pool"
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK", str(pool))
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_SIZE", "64M")
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_LOSS", "8")
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_REUSE", "1")
    server.cache_ram = 0
    server.start()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    request = {"prompt": prompt, "id_slot": 0, "n_predict": 1, "cache_prompt": True}
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    assert not list(pool.glob("*/meta.json"))  # a finished turn does not write

    if save_on_shutdown:
        process = server.process
        server.stop()
        assert process.returncode == 0
    else:
        switched = server.make_request("POST", "/completion", data={
            "prompt": "Something completely unrelated.", "id_slot": 0,
            "n_predict": 1, "cache_prompt": True,
        })
        assert switched.status_code == 200
    assert list(pool.glob("*/meta.json"))

    if save_on_shutdown:
        server.start()
    resumed = server.make_request("POST", "/completion", data=request)
    assert resumed.status_code == 200
    assert resumed.body["timings"]["prompt_n"] < first.body["timings"]["prompt_n"]

    # Automatic caching and the existing manual slot endpoints use the same format.
    saved = server.make_request("POST", "/slots/0?action=save", data={"filename": "native.bin"})
    assert saved.status_code == 200
    assert server.make_request("POST", "/slots/0?action=erase").status_code == 200
    restored = server.make_request("POST", "/slots/0?action=restore", data={"filename": "native.bin"})
    assert restored.status_code == 200
    assert restored.body["n_restored"] == saved.body["n_saved"]


def test_disk_prompt_cache_small_loss_skips_save(tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK", str(pool))
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_LOSS", "8192")
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_REUSE", "1")
    server.cache_ram = 0
    server.start()
    for prompt in ("A small cached conversation.", "An entirely different conversation."):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": 0, "n_predict": 1, "cache_prompt": True,
        })
        assert res.status_code == 200
        assert not list(pool.glob("*/meta.json"))


@pytest.fixture
def disk_pool(tmp_path, monkeypatch):
    pool = tmp_path / "pool"
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK", str(pool))
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_SIZE", "64M")
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_LOSS", "8")
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_REUSE", "1")
    return pool


@pytest.fixture
def disk_generated_server(disk_pool, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_LOSS", "32")
    server.cache_ram = 0
    server.n_slots = 1
    server.n_ctx = 1024
    return server


@pytest.mark.parametrize("restart", [False, True])
def test_disk_prompt_cache_generated_tail_skips_save(disk_generated_server, disk_pool, restart):
    if restart and os.name == "nt":
        pytest.skip("Popen.terminate uses TerminateProcess on Windows")
    server = disk_generated_server
    server.start()
    original = [10 + i % 50 for i in range(80)]

    def complete(tokens, n_predict):
        result = server.make_request("POST", "/completion", data={
            "prompt": tokens, "id_slot": 0, "n_predict": n_predict, "ignore_eos": True,
            "cache_prompt": True, "return_tokens": True,
        })
        assert result.status_code == 200
        assert len(result.body["tokens"]) == n_predict
        return result.body

    first = complete(original, 96)
    assert not list(disk_pool.glob("*/meta.json"))
    current_input = original
    if restart:
        server.stop()
        # Shutdown still saves the full state, including the generated tail.
        metadata = [json.loads(path.read_text()) for path in disk_pool.glob("*/meta.json")]
        assert any(item["n_tokens"] > len(original) + 32 for item in metadata)
        server.start()
        current_input = original + first["tokens"] + [90, 91]
        resumed = complete(current_input, 96)
        assert resumed["timings"]["cache_n"] >= len(original)

    before = set(disk_pool.glob("*/meta.json"))
    # A template clearing reasoning presents the same input prefix without the generated tail.
    cleared = complete(current_input + [94, 95], 1)
    assert cleared["timings"]["cache_n"] >= len(current_input) - 1
    assert set(disk_pool.glob("*/meta.json")) == before


def test_disk_prompt_cache_generated_text_becomes_input(disk_generated_server, disk_pool):
    server = disk_generated_server
    server.start()
    original = [10 + i % 50 for i in range(80)]
    first = server.make_request("POST", "/completion", data={
        "prompt": original, "id_slot": 0, "n_predict": 64, "ignore_eos": True,
        "cache_prompt": True, "return_tokens": True,
    })
    assert first.status_code == 200
    assert len(first.body["tokens"]) == 64
    continued = original + first.body["tokens"] + [90, 91]
    result = server.make_request("POST", "/completion", data={
        "prompt": continued, "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert result.status_code == 200
    assert not list(disk_pool.glob("*/meta.json"))
    # The generated text is now part of a request's input, so editing it counts toward loss.
    result = server.make_request("POST", "/completion", data={
        "prompt": original + [94, 95], "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert result.status_code == 200
    assert list(disk_pool.glob("*/meta.json"))


def test_disk_prompt_cache_cancelled_generated_tail(disk_generated_server, disk_pool):
    server = disk_generated_server
    server.n_ctx = 4096
    server.server_slots = True
    server.start()
    original = [10 + i % 50 for i in range(80)]
    with requests.post(server.make_url("/completion"), json={
        "prompt": original, "id_slot": 0, "n_predict": -1, "ignore_eos": True,
        "cache_prompt": True, "return_tokens": True, "stream": True,
    }, stream=True, timeout=30) as response:
        assert response.status_code == 200
        generated = 0
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b"data: "):
                generated += len(json.loads(line[6:]).get("tokens", []))
                if generated >= 64:
                    break
        assert generated >= 64
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        slots = server.make_request("GET", "/slots")
        assert slots.status_code == 200
        if not slots.body[0]["is_processing"]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("cancelled request did not release its slot")
    result = server.make_request("POST", "/completion", data={
        "prompt": original + [94, 95], "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert result.status_code == 200
    assert result.body["timings"]["cache_n"] >= len(original)
    assert not list(disk_pool.glob("*/meta.json"))


def test_disk_prompt_cache_generated_tail_after_context_shift(disk_generated_server, disk_pool):
    server = disk_generated_server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    original = [10 + i % 50 for i in range(160)]
    first = server.make_request("POST", "/completion", data={
        "prompt": original, "id_slot": 0, "n_predict": 160, "ignore_eos": True,
        "n_keep": 15, "n_discard": 96, "cache_prompt": True, "return_tokens": True,
    })
    assert first.status_code == 200
    assert len(first.body["tokens"]) == 160
    assert first.body["truncated"]
    # n_keep includes one extra BOS position: [16, 112) was removed, leaving 64 input tokens.
    retained_input = original[:16] + original[112:]
    result = server.make_request("POST", "/completion", data={
        "prompt": retained_input + [94, 95], "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert result.status_code == 200
    assert result.body["timings"]["cache_n"] >= len(retained_input)
    assert not list(disk_pool.glob("*/meta.json"))


def test_disk_prompt_cache_multimodal_generated_tail(mmproj_server, disk_pool, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CACHE_DISK_MIN_LOSS", "32")
    server = mmproj_server
    server.cache_ram = 0
    server.n_slots = 1
    server.n_ctx = 1024
    server.swa_full = True
    server.n_ubatch = 512
    server.start()
    image = _get_img_base64(IMG_URL_CAT)

    def complete(suffix, n_predict):
        result = server.make_request("POST", "/completion", data={
            "prompt": {"prompt_string": "Describe this: <__media__>\n" + suffix, "multimodal_data": [image]},
            "id_slot": 0, "n_predict": n_predict, "ignore_eos": True, "cache_prompt": True,
        })
        assert result.status_code == 200
        return result.body

    first = complete("", 64)
    assert first["timings"]["predicted_n"] == 64
    cleared = complete("Now summarize it.", 1)
    assert cleared["timings"]["cache_n"] >= first["timings"]["prompt_n"] - 8
    assert not list(disk_pool.glob("*/meta.json"))


def test_disk_prompt_cache_sleep(disk_pool):
    server.cache_ram = 0
    server.sleep_idle_seconds = 1
    server.start()
    request = {"prompt": "The quick brown fox jumps over the lazy dog. " * 8,
               "id_slot": 0, "n_predict": 1, "cache_prompt": True}
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        props = server.make_request("GET", "/props")
        assert props.status_code == 200
        if props.body["is_sleeping"] and list(disk_pool.glob("*/meta.json")):
            break
        time.sleep(0.1)
    else:
        pytest.fail("server did not flush its disk cache on sleep")
    resumed = server.make_request("POST", "/completion", data=request)
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] == first.body["timings"]["prompt_n"] - 1
    assert resumed.body["content"] == first.body["content"]


def test_disk_prompt_cache_request_opt_out(disk_pool):
    server.cache_ram = 0
    server.start()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    request = {"prompt": prompt, "id_slot": 0, "n_predict": 1, "cache_prompt": False}
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    server.stop()
    assert not list(disk_pool.glob("*/meta.json"))
    server.start()
    resumed = server.make_request("POST", "/completion", data=request)
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] == 0
    assert resumed.body["timings"]["prompt_n"] == first.body["timings"]["prompt_n"]


@pytest.mark.parametrize("setting", ["kv-type", "flash-attention"])
def test_disk_prompt_cache_separates_context_settings(disk_pool, setting):
    server.cache_ram = 0
    server.fa = "on"
    server.start()
    request = {"prompt": "The quick brown fox jumps over the lazy dog. " * 8,
               "id_slot": 0, "n_predict": 1, "cache_prompt": True}
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    server.stop()
    assert list(disk_pool.glob("*/meta.json"))
    if setting == "kv-type":
        server.ctk = "f32"
    else:
        server.fa = "off"
    server.start()
    resumed = server.make_request("POST", "/completion", data=request)
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] == 0
    assert resumed.body["timings"]["prompt_n"] == first.body["timings"]["prompt_n"]


@pytest.mark.skipif(os.name == "nt", reason="Popen.terminate uses TerminateProcess on Windows")
def test_disk_prompt_cache_shutdown_during_generation(disk_pool):
    server.cache_ram = 0
    server.n_predict = -1
    server.n_ctx = 2048
    server.enable_ctx_shift = True
    server.start()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8
    tokenized = server.make_request("POST", "/tokenize", data={"content": prompt, "add_special": True})
    assert tokenized.status_code == 200
    with requests.post(server.make_url("/completion"), json={
        "prompt": prompt, "id_slot": 0, "n_predict": -1, "ignore_eos": True,
        "cache_prompt": True, "stream": True,
    }, stream=True, timeout=30) as response:
        assert response.status_code == 200
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b"data: "):
                assert not json.loads(line[6:]).get("stop", False)
                break
        else:
            pytest.fail("generation did not produce a streamed token")
        process = server.process
        assert process is not None
        process.terminate()
        process.wait(timeout=30)
        assert process.returncode == 0
    server.stop()
    assert list(disk_pool.glob("*/meta.json"))
    server.start()
    resumed = server.make_request("POST", "/completion", data={
        "prompt": prompt, "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert resumed.status_code == 200
    assert resumed.body["timings"]["cache_n"] == len(tokenized.body["tokens"]) - 1


@pytest.mark.parametrize("restart", [False, True])
def test_disk_prompt_cache_draft_state(disk_pool, restart):
    server.cache_ram = 0
    server.model_draft_hf_repo = server.model_hf_repo
    server.spec_type = "draft-simple"
    server.start()
    prompt = "The quick brown fox jumps over the lazy dog. " * 8

    def complete(text, slot, n_predict=16):
        result = server.make_request("POST", "/completion", data={
            "prompt": text, "id_slot": slot, "n_predict": n_predict, "cache_prompt": True,
        })
        assert result.status_code == 200
        return result.body

    original = complete(prompt, 0)
    assert complete(prompt, 1)["content"] == original["content"]
    complete("An unrelated conversation replaces this slot.", 0, 1)
    continuation = prompt + original["content"] + " And"
    direct = complete(continuation, 1, 32)
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    if restart:
        server.stop()
        server.start()
    resumed = complete(continuation, 0, 32)
    assert resumed["content"] == direct["content"]
    for metric in ("cache_n", "prompt_n", "draft_n", "draft_n_accepted"):
        assert resumed["timings"][metric] == direct["timings"][metric]
    assert resumed["timings"]["draft_n_accepted"] > 0


@pytest.mark.parametrize("restart", [False, True])
def test_disk_prompt_cache_restored_checkpoint(swa_window_server, disk_pool, restart, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT", "32")
    server = swa_window_server
    server.n_ubatch = 32
    server.cache_ram = 0
    server.start()
    base = "The quick brown fox jumps over the lazy dog. " * 40
    original = base + "The first ending of this story is a happy one."
    edited = "The quick brown fox jumps over the lazy dog. " * 15 + "But the second ending was different and sad."

    def complete(prompt, slot):
        result = server.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": slot, "n_predict": 1,
            "cache_prompt": True, "return_tokens": True,
        })
        assert result.status_code == 200
        return result.body

    # Keep checkpoints from an earlier turn, then extend well past the SWA window.
    warmup = "The quick brown fox jumps over the lazy dog. " * 12
    complete(warmup, 0)
    first = complete(original, 0)
    complete(warmup, 1)
    complete(original, 1)
    complete("An unrelated chat replaces the first slot.", 0)
    metadata = [json.loads(path.read_text()) for path in disk_pool.glob("*/meta.json")]
    assert any(entry["checkpoints"] for entry in metadata)

    direct = complete(edited, 1)
    assert 0 < direct["timings"]["cache_n"] < first["timings"]["prompt_n"] - 16
    # Do not let the baseline's new ending become a shortcut in the restart case.
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    if restart:
        server.stop()
        server.start()
    restored = complete(edited, 0)
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["tokens"] == direct["tokens"]


def test_disk_prompt_cache_invalid_state_falls_back(disk_pool):
    server.cache_ram = 0
    server.start()
    request = {
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    }
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    switched = server.make_request("POST", "/completion", data={
        "prompt": "An unrelated conversation replaces this slot.", "id_slot": 0,
        "n_predict": 1, "cache_prompt": True,
    })
    assert switched.status_code == 200
    entries = list(disk_pool.glob("*/meta.json"))
    assert len(entries) == 1
    entry = entries[0].parent
    # Keep the length valid, but invalidate the native state header.
    with (entry / "state.bin").open("r+b") as state:
        state.write(b"BAD!")
    restored = server.make_request("POST", "/completion", data=request)
    assert restored.status_code == 200
    assert restored.body["content"] == first.body["content"]
    # The invalid snapshot is rejected before any restore, so the live slot is untouched and
    # only its shared BOS can be reused.
    assert restored.body["timings"]["cache_n"] <= 1
    assert restored.body["timings"]["prompt_n"] >= first.body["timings"]["prompt_n"] - 1
    assert not entry.exists()


def test_disk_prompt_cache_failed_restore_keeps_snapshot(disk_pool):
    server.cache_ram = 0
    server.start()
    request = {
        "prompt": "The quick brown fox jumps over the lazy dog. " * 8,
        "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    }
    first = server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    switched = server.make_request("POST", "/completion", data={
        "prompt": "An unrelated conversation replaces this slot.", "id_slot": 0,
        "n_predict": 1, "cache_prompt": True,
    })
    assert switched.status_code == 200
    entries = list(disk_pool.glob("*/meta.json"))
    assert len(entries) == 1
    entry = entries[0].parent
    # Valid header and length, but a corrupt payload: the native restore fails.
    with (entry / "state.bin").open("r+b") as state:
        state.seek(STATE_FILE_HEADER_SIZE)
        state.write(b"\xff" * 64)
    restored = server.make_request("POST", "/completion", data=request)
    assert restored.status_code == 200
    assert restored.body["content"] == first.body["content"]
    assert restored.body["timings"]["prompt_n"] == first.body["timings"]["prompt_n"]
    assert entry.exists()


def test_disk_prompt_cache_uses_rendered_template_kwargs(disk_pool, tmp_path):
    template = tmp_path / "cache-template.jinja"
    template.write_text(
        "{{ marker | default('default') }}\n"
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    server.jinja = True
    server.chat_template_file = str(template)
    server.cache_ram = 0
    server.start()
    original = "The quick brown fox jumps over the lazy dog. " * 8

    def chat(content, kwargs):
        result = server.make_request("POST", "/v1/chat/completions", data={
            "messages": [{"role": "user", "content": content}], "max_tokens": 1,
            "chat_template_kwargs": kwargs, "temperature": 0.0,
        })
        assert result.status_code == 200
        return result.body

    first = chat(original, {"marker": "A"})
    chat("A completely unrelated conversation.", {"marker": "A"})
    resumed = chat(original, {"marker": "A", "unused_argument": True})
    assert resumed["timings"]["cache_n"] == first["timings"]["prompt_n"] - 1
    changed = chat(original, {"marker": "B"})
    assert changed["timings"]["cache_n"] < 16
    assert changed["timings"]["prompt_n"] > first["timings"]["prompt_n"] // 2


@pytest.mark.parametrize("architecture", ["qwen35", "mamba"])
@pytest.mark.parametrize("restart", [False, True])
def test_disk_prompt_cache_recurrent_checkpoint(disk_pool, tmp_path, monkeypatch, architecture, restart):
    # Generate these with test-llama-archs -a '^(qwen35|mamba)$' -o DIR.
    directory = os.environ.get("LLAMA_TEST_RECURRENT_MODELS")
    if not directory:
        pytest.skip("set LLAMA_TEST_RECURRENT_MODELS to generated test-models directory")
    model = Path(directory) / f"{architecture}-dense.gguf"
    if not model.is_file():
        pytest.skip(f"generated test model missing: {model}")
    monkeypatch.setenv("LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT", "16")
    recurrent = ServerProcess()
    recurrent.model_file = str(model)
    recurrent.model_hf_repo = None
    recurrent.model_hf_file = None
    recurrent.model_alias = architecture
    recurrent.n_ctx = 512
    recurrent.n_slots = 2
    recurrent.n_batch = 64
    recurrent.n_ubatch = 16
    recurrent.n_threads = 2
    recurrent.temperature = 0.0
    recurrent.cache_ram = 0
    recurrent.slot_save_path = str(tmp_path)
    recurrent.start()

    def complete(prompt, slot):
        result = recurrent.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": slot, "n_predict": 1,
            "cache_prompt": True, "return_tokens": True,
        })
        assert result.status_code == 200
        return result.body

    original = [10 + i % 50 for i in range(180)]
    edited = original[:80] + [90 + i for i in range(10)]
    complete(original[:72], 0)
    complete(original, 0)
    complete(original[:72], 1)
    complete(original, 1)
    complete([100] * 20, 0)
    direct = complete(edited, 1)
    assert 0 < direct["timings"]["cache_n"] < 80
    assert recurrent.make_request("POST", "/slots/1?action=erase").status_code == 200
    if restart:
        recurrent.stop()
        recurrent.start()
    restored = complete(edited, 0)
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["tokens"] == direct["tokens"]


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("change_image", [False, True])
@pytest.mark.parametrize("mmproj_server", ["tinygemma3", "causal", "causal-unmixed"], indirect=True)
def test_disk_prompt_cache_multimodal(mmproj_server, disk_pool, restart, change_image):
    server = mmproj_server
    server.swa_full = True
    server.cache_ram = 0
    server.n_ubatch = 512
    server.start()
    cat = _get_img_base64(IMG_URL_CAT)
    truck = _get_img_base64(IMG_URL_TRUCK)

    def complete(image, slot):
        result = server.make_request("POST", "/completions", data={
            "prompt": {"prompt_string": "What is this: <__media__>\n", "multimodal_data": [image]},
            "id_slot": slot, "cache_prompt": True, "temperature": 0.0, "top_k": 1,
        })
        assert result.status_code == 200
        return result.body

    first = complete(cat, 0)
    complete(cat, 1)
    switched = server.make_request("POST", "/completion", data={
        "prompt": "An unrelated plain text conversation.", "id_slot": 0, "n_predict": 1, "cache_prompt": True,
    })
    assert switched.status_code == 200
    assert list(disk_pool.glob("*/meta.json"))
    image = truck if change_image else cat
    direct = complete(image, 1)
    assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
    if restart:
        server.stop()
        server.start()
    restored = complete(image, 0)
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["content"] == direct["content"]
    if change_image:
        assert restored["timings"]["cache_n"] < 16
    else:
        assert restored["timings"]["cache_n"] == first["timings"]["prompt_n"] - 1


def test_disk_prompt_cache_multimodal_text_prefix(mmproj_server, disk_pool):
    server = mmproj_server
    server.swa_full = True
    server.cache_ram = 0
    server.n_ubatch = 512
    server.start()
    prefix = "The quick brown fox jumps over the lazy dog. " * 4
    prompt = {"prompt_string": prefix + "<__media__>\n", "multimodal_data": [_get_img_base64(IMG_URL_CAT)]}

    def complete(text, slot):
        result = server.make_request("POST", "/completion", data={
            "prompt": text, "id_slot": slot, "n_predict": 1, "cache_prompt": True,
        })
        assert result.status_code == 200
        return result.body

    complete(prompt, 0)
    complete(prompt, 1)
    complete("An unrelated conversation replaces the slot.", 0)
    edited = prefix + "A fresh ending without an image."
    direct = complete(edited, 1)
    restored = complete(edited, 0)
    assert restored["timings"]["cache_n"] > 30
    assert restored["timings"]["cache_n"] == direct["timings"]["cache_n"]
    assert restored["timings"]["prompt_n"] == direct["timings"]["prompt_n"]
    assert restored["content"] == direct["content"]


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join(server.slot_save_path, "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed


# the slot file carries the live draft state: a restored slot drafts like a slot that was never saved
def test_slot_restore_draft_state():
    global server
    server.model_draft_hf_repo = server.model_hf_repo
    server.spec_type = "draft-simple"
    server.start()

    prompt = "What is the capital of France?"
    turn = {}
    for id_slot in [0, 1]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 16,
            "id_slot": id_slot,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        turn[id_slot] = res.body["content"]
    assert turn[0] == turn[1]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "draft_slot0.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "draft_slot0.bin",
    })
    assert res.status_code == 200

    timings = {}
    for id_slot in [0, 1]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt + turn[id_slot] + " And",
            "n_predict": 32,
            "id_slot": id_slot,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        timings[id_slot] = res.body["timings"]

    assert timings[0]["prompt_n"] == timings[1]["prompt_n"]
    assert timings[0]["draft_n_accepted"] > 0
    assert timings[0]["draft_n"]          == timings[1]["draft_n"]
    assert timings[0]["draft_n_accepted"] == timings[1]["draft_n_accepted"]


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server(request, monkeypatch):
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    kind = getattr(request, "param", "tinygemma3")
    if kind.startswith("causal"):
        model = os.environ.get("LLAMA_TEST_CAUSAL_MM_MODEL")
        projector = os.environ.get("LLAMA_TEST_CAUSAL_MM_PROJECTOR")
        if not model or not projector:
            pytest.skip("set LLAMA_TEST_CAUSAL_MM_MODEL and LLAMA_TEST_CAUSAL_MM_PROJECTOR to a causal vision model")
        mm_server = ServerProcess()
        mm_server.model_file = model
        mm_server.model_hf_repo = None
        mm_server.model_hf_file = None
        mm_server.model_alias = "causal-vision-test"
        mm_server.n_ctx = 2048
        mm_server.n_batch = 512
        mm_server.n_slots = 2
        mm_server.n_predict = 4
        mm_server.n_threads = 2
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", projector)
        monkeypatch.setenv("LLAMA_SERVER_MTMD_MIXED", "0" if kind == "causal-unmixed" else "1")
    else:
        mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


@pytest.fixture
def swa_server():
    swa = ServerPreset.tinygemma3()
    swa.slot_save_path = "./tmp"
    swa.temperature = 0.0
    swa.cache_ram = 0
    # Keep the first prompt checkpoint before the divergence point.
    swa.n_ubatch = 32
    return swa


# the non-ASCII name checks that the appendix lands in the same file as the llama state on Windows
@pytest.mark.parametrize("filename", ["ckpt_slot1.bin", "ckpt_slot1_é.bin"])
def test_slot_restore_preserves_context_checkpoints(swa_server, filename):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]
    assert n_live < n_full

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0
    ckpt_file = os.path.join(server.slot_save_path, filename)
    assert res.body["n_written"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": "Unrelated text with no common prefix occupies the slot now.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_read"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


# checkpoint appendix: magic(4) version(4) count(4), then per checkpoint
# n_tokens(8) pos_min(4) pos_max(4) and three blobs (target, draft, speculative), each size(8) + data
def parse_ckpt_appendix(data):
    off = data.find(struct.pack("<II", 0x504b4353, 1))
    assert off > 0
    count = struct.unpack_from("<I", data, off + 8)[0]
    ckpts = []
    pos = off + 12
    for _ in range(count):
        start = pos
        pos += 16
        blobs = []
        for _ in range(3):
            n = struct.unpack_from("<Q", data, pos)[0]
            blobs.append(pos + 8)
            pos += 8 + n
        ckpts.append((start, pos, blobs[0]))
    assert pos == len(data)
    return off, ckpts


# a damaged appendix must be ignored, or its checkpoints dropped when they fail to load, without aborting the server
@pytest.mark.parametrize("damage", ["oversized_blob", "empty_target", "corrupt_state", "many_checkpoints"])
def test_slot_restore_damaged_checkpoint_appendix(swa_server, damage):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "ckpt_damaged.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    off, ckpts = parse_ckpt_appendix(data)

    if damage == "oversized_blob":
        # the first target blob declares a size that cannot be allocated, it must be rejected before allocating
        data = data[:ckpts[0][0] + 16] + struct.pack("<Q", 1 << 62)
    elif damage == "empty_target":
        # the target blobs are removed and their size set to 0, a valid save never writes an empty target state
        for start, end, tgt in reversed(ckpts):
            size = struct.unpack_from("<Q", data, tgt - 8)[0]
            data = data[:tgt - 8] + struct.pack("<Q", 0) + data[tgt + size:]
    elif damage == "corrupt_state":
        # the sizes are intact, but the target states do not load
        for _, _, tgt in ckpts:
            struct.pack_into("<I", data, tgt, 0xdeadbeef)
    else:
        # more than 1024 entries: one-byte fillers that never match go first, the real checkpoints stay last
        filler = struct.pack("<qiiQBQQ", 0, 0, 1 << 30, 1, 0, 0, 0)
        data = data[:off + 12] + filler * (1025 - len(ckpts)) + data[off + 12:]
        struct.pack_into("<I", data, off + 8, 1025)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200
    if damage in ("oversized_blob", "empty_target"):
        assert res.body["n_read"] == off

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    if damage == "many_checkpoints":
        assert res.body["timings"]["prompt_n"] == n_live
    else:
        assert res.body["timings"]["prompt_n"] > n_live


# the draft blobs of the checkpoint appendix are not covered by the main payload checks,
# so restoring into a server with another draft KV cache type must not abort
@pytest.mark.parametrize("ctkd_restore", ["f16", "q8_0"])
def test_slot_restore_checkpoints_draft_kv_type_change(swa_server, ctkd_restore):
    server = swa_server
    server.model_draft_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"  # same file as the target, already in the HF cache
    server.spec_type = "draft-simple"
    server.ctkd = "f16"
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    server.stop()
    server.ctkd = ctkd_restore
    server.start()

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


# a SWA window smaller than the prompt, so a saved state holds only the cells inside the window
@pytest.fixture
def swa_window_server(tmp_path):
    swa = ServerPreset.tinygemma3()
    swa.override_kv = "gemma3.attention.sliding_window=int:64"
    swa.slot_save_path = str(tmp_path)
    swa.temperature = 0.0
    swa.n_predict = 8
    return swa


SWA_PROMPT = "The quick brown fox jumps over the lazy dog. " * 30


def tokenize(server, text):
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return res.body["tokens"]


# the first turn and the exact token continuation of it, so the next turn extends the saved state
def swa_turns(server, id_slot):
    res = server.make_request("POST", "/completion", data={
        "prompt": tokenize(server, SWA_PROMPT),
        "id_slot": id_slot,
        "cache_prompt": True,
        "return_tokens": True,
    })
    assert res.status_code == 200
    return tokenize(server, SWA_PROMPT) + res.body["tokens"] + tokenize(server, " The end.")


def swa_next(server, id_slot, prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]


# a restored SWA state holds the window the next token needs, so it is reused as is
def test_slot_restore_swa_window(swa_window_server):
    server = swa_window_server
    server.cache_ram = 0
    server.start()

    prompt = {id_slot: swa_turns(server, id_slot) for id_slot in [0, 1]}
    assert prompt[0] == prompt[1]

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "swa_slot0.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "swa_slot0.bin"})
    assert res.status_code == 200

    timings = {id_slot: swa_next(server, id_slot, prompt[id_slot]) for id_slot in [0, 1]}

    assert timings[0]["cache_n"]  == timings[1]["cache_n"]
    assert timings[0]["prompt_n"] == timings[1]["prompt_n"]


# a SWA state loaded back from the RAM prompt cache is reused as is
def test_prompt_cache_swa_window(swa_window_server):
    server = swa_window_server
    server.n_slots = 1

    def run(other_prompt_first):
        server.start()
        prompt = swa_turns(server, 0)
        if other_prompt_first:
            res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time " * 60, "cache_prompt": True})
            assert res.status_code == 200
        timings = swa_next(server, 0, prompt)
        server.stop()
        return timings

    direct = run(False)
    cached = run(True)

    assert cached["cache_n"]  == direct["cache_n"]
    assert cached["prompt_n"] == direct["prompt_n"]
