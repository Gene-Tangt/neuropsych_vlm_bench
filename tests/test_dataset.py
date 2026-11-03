import pytest

from neuropsych_bench import TaskLoader, load_rows, task_names
from neuropsych_bench.dataset import DATA_ROOT

# Every task, low then mid then high stage
TASK_ORDER = [
    "borb_orientation", "borb_line_length_comparison", "borb_size_comparison", "borb_position_of_gap",
    "mindset_weber_law",
    "borb_overlapped_letters", "borb_overlapped_shapes", "left_embedded_figures", "mindset_2D_transformations",
    "mindset_NAP_vs_MP_2D_lines", "mindset_NAP_vs_MP_3D_geons", "mindset_amodal_completion", "mindset_decomposition",
    "mindset_dotted_linedrawings", "mindset_geometric_elements_comparison",
    "mindset_global_change_linedrawings_fragmented", "mindset_global_change_linedrawings_frankenstein",
    "mindset_global_change_silhouettes_fragmented", "mindset_global_change_silhouettes_frankenstein",
    "mindset_relational_vs_coordinate", "mindset_segmented_images", "mindset_texturized_blobs",
    "mindset_texturized_linedrawings_chars", "mindset_texturized_linedrawings_lines",
    "borb_triplet_letters", "borb_triplet_shapes", "mindset_global_change_linedrawings_original",
    "mindset_global_change_silhouettes_original", "mindset_linedrawings", "mindset_silhouettes",
    "mindset_viewpoint_invariance",
]
LAYOUTS = {("target",), ("options", "options"), ("target", "option_1", "option_2"),
           ("target", "option_1", "option_2", "option_3")}
LIST_ANSWER_TASKS = {"borb_overlapped_shapes", "borb_triplet_shapes"}
# Every row has exactly these fields, with these types
SCHEMA = {"index": int, "id": str, "stage": str, "process": str, "task": str, "task_type": str,
          "num_images": int, "prompt": str, "images": list, "image_roles": list, "answer": (str, list)}


def test_rows():
    rows = load_rows()
    assert len(rows) == 1130
    for r in rows:
        assert set(r) == set(SCHEMA), (r.get("id"), set(r) ^ set(SCHEMA))
        assert all(isinstance(r[k], t) for k, t in SCHEMA.items()), r["id"]
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["id"].startswith(r["task"] + "/")
        assert len(r["images"]) == len(r["image_roles"]) == r["num_images"]
        assert tuple(r["image_roles"]) in LAYOUTS
        assert r["stage"] in ("low", "mid", "high")
        assert "{}" in r["prompt"]


def test_answer_types():
    for r in load_rows():
        if r["task"] in LIST_ANSWER_TASKS:
            assert isinstance(r["answer"], list) and all(isinstance(a, str) for a in r["answer"]), r["id"]
        else:
            assert isinstance(r["answer"], str) and r["answer"], r["id"]


def test_task_metadata_constant_within_task():
    for task in task_names():
        rows = TaskLoader(task).rows
        for key in ("stage", "process", "task_type", "num_images"):
            assert len({r[key] for r in rows}) == 1, (task, key)


def test_task_order():
    assert task_names() == TASK_ORDER


def test_task_loader():
    loader = TaskLoader("left_embedded_figures")
    info = loader.get_task_info()
    assert info == {"stage": "mid", "process": "figure_judgement", "task": "left_embedded_figures",
                    "task_type": "match_to_sample", "num_images": 4}
    trial = loader.get_trials()[0]
    assert trial["trial_id"] == "trial_000" and trial["answer_key"] == "3"
    assert trial["data_root"] == DATA_ROOT
    assert len(loader.get_trials()) == len(loader.rows)


def test_unknown_task():
    with pytest.raises(ValueError):
        TaskLoader("not_a_task")


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main(["-v", __file__]))
