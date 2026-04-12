from collections.abc import Iterator

import pytest

from aimu.prompts import Prompt, PromptCatalog


@pytest.fixture
def prompt_catalog(tmp_path) -> Iterator[PromptCatalog]:
    with PromptCatalog(str(tmp_path / "test.db")) as catalog:
        yield catalog


def test_store_prompt(prompt_catalog):
    name = "task"
    model_id = "x"

    prompt1 = Prompt(name=name, prompt="test", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)
    prompt2 = prompt_catalog.retrieve_last(name, model_id)

    assert prompt1.id == prompt2.id


def test_auto_versioning(prompt_catalog):
    name = "task"
    model_id = "x"

    p1 = Prompt(name=name, prompt="a", model_id=model_id)
    p2 = Prompt(name=name, prompt="b", model_id=model_id)
    p3 = Prompt(name=name, prompt="c", model_id=model_id)
    prompt_catalog.store_prompt(p1)
    prompt_catalog.store_prompt(p2)
    prompt_catalog.store_prompt(p3)

    assert p1.version == 1
    assert p2.version == 2
    assert p3.version == 3


def test_created_at_set_automatically(prompt_catalog):
    p = Prompt(name="task", prompt="hello", model_id="x")
    prompt_catalog.store_prompt(p)
    assert p.created_at is not None


def test_metrics_stored_as_json(prompt_catalog):
    metrics = {"accuracy": 0.94, "precision": 0.91, "recall": 0.97, "n_samples": 120}
    p = Prompt(name="task", prompt="hello", model_id="x", metrics=metrics)
    prompt_catalog.store_prompt(p)

    retrieved = prompt_catalog.retrieve_last("task", "x")
    assert retrieved.metrics == metrics


def test_retrieve_last(prompt_catalog):
    name = "task"
    model_id = "x"

    prompt1 = Prompt(name=name, prompt="a", model_id=model_id, version=1)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(name=name, prompt="c", model_id=model_id, version=3)
    prompt_catalog.store_prompt(prompt2)

    prompt3 = Prompt(name=name, prompt="b", model_id=model_id, version=2)
    prompt_catalog.store_prompt(prompt3)

    prompt4 = prompt_catalog.retrieve_last(name, model_id)

    assert prompt2.id == prompt4.id


def test_retrieve_last_isolated_by_name(prompt_catalog):
    model_id = "x"

    prompt_catalog.store_prompt(Prompt(name="task_a", prompt="a1", model_id=model_id))
    prompt_catalog.store_prompt(Prompt(name="task_a", prompt="a2", model_id=model_id))
    prompt_catalog.store_prompt(Prompt(name="task_b", prompt="b1", model_id=model_id))

    last_a = prompt_catalog.retrieve_last("task_a", model_id)
    last_b = prompt_catalog.retrieve_last("task_b", model_id)

    assert last_a.prompt == "a2"
    assert last_b.prompt == "b1"


def test_retrieve_all(prompt_catalog):
    name = "task"
    model_id = "x"

    prompt1 = Prompt(name=name, prompt="a", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(name=name, prompt="c", model_id=model_id)
    prompt_catalog.store_prompt(prompt2)

    prompts = prompt_catalog.retrieve_all(name, model_id)

    assert prompt1 in prompts
    assert prompt2 in prompts


def test_delete_all(prompt_catalog):
    name = "task"
    model_id = "x"

    prompt1 = Prompt(name=name, prompt="a", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)

    prompt2 = Prompt(name=name, prompt="c", model_id=model_id)
    prompt_catalog.store_prompt(prompt2)

    prompts = prompt_catalog.retrieve_all(name, model_id)

    assert prompt1 in prompts
    assert prompt2 in prompts

    rows_deleted = prompt_catalog.delete_all(name, model_id)

    assert rows_deleted == 2

    prompts = prompt_catalog.retrieve_all(name, model_id)

    assert len(prompts) == 0


def test_retrieve_model_ids(prompt_catalog):
    model_id = "x"

    prompt1 = Prompt(name="task", prompt="a", model_id=model_id)
    prompt_catalog.store_prompt(prompt1)

    model_ids = prompt_catalog.retrieve_model_ids()

    assert len(model_ids) == 1
    assert model_ids[0] == "x"


def test_retrieve_names(prompt_catalog):
    model_id = "x"

    prompt_catalog.store_prompt(Prompt(name="classifier", prompt="a", model_id=model_id))
    prompt_catalog.store_prompt(Prompt(name="summarizer", prompt="b", model_id=model_id))
    prompt_catalog.store_prompt(Prompt(name="classifier", prompt="c", model_id=model_id))

    names = prompt_catalog.retrieve_names()

    assert set(names) == {"classifier", "summarizer"}
