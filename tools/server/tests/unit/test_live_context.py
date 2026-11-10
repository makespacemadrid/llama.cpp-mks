import pytest
from utils import *

# POST /props {"context": {...}} replaces the llama_context in place, keeping the
# model weights: the KV pool size, parallel slots and batch sizes change without
# reloading the server.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 2


def complete(n_predict: int = 8):
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": n_predict,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    return res.body


def live_props():
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body


def test_context_change_requires_props_endpoint():
    server.start()
    res = server.make_request("POST", "/props", data={"context": {"n_ctx": 256}})
    assert res.status_code == 501


def test_shrink_and_grow_back_keep_serving():
    server.server_props = True
    server.start()
    before = complete()

    res = server.make_request("POST", "/props", data={"context": {"n_ctx": 256, "n_parallel": 1}})
    assert res.status_code == 200
    assert res.body["context"]["n_ctx"] == 256
    assert res.body["context"]["n_parallel"] == 1
    assert res.body["context"]["n_ctx_slot"] == 256
    assert live_props()["default_generation_settings"]["n_ctx"] == 256
    # same weights, greedy sampling: the output does not depend on the pool size
    assert complete()["content"] == before["content"]

    res = server.make_request("POST", "/props", data={"context": {"n_ctx": 512, "n_parallel": 2}})
    assert res.status_code == 200
    assert res.body["context"]["n_ctx"] == 512
    assert res.body["context"]["n_parallel"] == 2
    assert complete()["content"] == before["content"]


def test_batch_sizes_change_live():
    server.server_props = True
    server.start()
    res = server.make_request("POST", "/props", data={"context": {"n_ubatch": 32, "n_batch": 64}})
    assert res.status_code == 200
    assert res.body["context"]["n_ubatch"] == 32
    assert res.body["context"]["n_batch"] == 64
    assert res.body["context"]["n_ctx"] == 512  # unspecified fields keep their value
    complete()


@pytest.mark.parametrize("context", [
    {"n_ctx": 0},
    {"n_ctx": -1},
    {"n_ctx": "256"},
    {"n_parallel": 1.5},
])
def test_invalid_context_values_are_rejected(context):
    server.server_props = True
    server.start()
    res = server.make_request("POST", "/props", data={"context": context})
    assert res.status_code == 400
    complete()


def test_context_must_be_an_object():
    server.server_props = True
    server.start()
    res = server.make_request("POST", "/props", data={"context": 256})
    assert res.status_code == 400
