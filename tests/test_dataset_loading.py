from rgc_llm.training.dataset import load_records_from_path


def test_jsonl_loading(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"input": "a", "target": "b"}\n{"input": "c", "target": "d"}\n', encoding="utf-8")
    data = load_records_from_path(path)
    assert len(data) == 2
    assert data[0].input == "a"


def test_json_loading(tmp_path):
    path = tmp_path / "train.json"
    path.write_text('[{"input": "x", "target": "y"}]', encoding="utf-8")
    data = load_records_from_path(path)
    assert len(data) == 1
    assert data[0].target == "y"
