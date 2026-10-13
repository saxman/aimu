import json

import anthropic
from pydantic import BaseModel

from aimu.models._internal.json_safe import json_safe


def test_plain_json_passes_through_unchanged():
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "stream": False}
    assert json_safe(payload) == payload


def test_not_given_keys_are_dropped_because_the_sdk_omits_them_from_the_wire():
    assert json_safe({"model": "m", "system": anthropic.NOT_GIVEN, "tools": anthropic.NOT_GIVEN}) == {"model": "m"}


def test_a_pydantic_instance_becomes_its_json_form():
    class Point(BaseModel):
        x: int

    assert json_safe({"p": Point(x=1)}) == {"p": {"x": 1}}


def test_a_class_becomes_a_visible_marker():
    class Verdict(BaseModel):
        ok: bool

    out = json_safe({"response_format": Verdict})
    assert out["response_format"].startswith("<class ") and "Verdict" in out["response_format"]


def test_bytes_and_unknown_objects_become_visible_markers_not_reprs():
    out = json_safe({"raw": b"\x00\x01\x02", "obj": object(), "tuple": (1, 2)})
    assert out["raw"] == "<bytes 3>"
    assert out["obj"] == "<builtins.object>"
    assert out["tuple"] == [1, 2]
    json.dumps(out)


def test_non_string_dict_keys_are_stringified():
    assert json_safe({1: "a"}) == {"1": "a"}
