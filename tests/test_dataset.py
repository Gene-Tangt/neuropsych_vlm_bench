from neuropsych_bench import TaskLoader, load_rows, task_names
from neuropsych_bench.dataset import DATA_ROOT

# Task order of the old test_specs/test_list.json (low, mid, high)
OLD_ORDER = [
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


def test_rows():
    rows = load_rows()
    assert len(rows) == 1130
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["id"].startswith(r["task"] + "/")
        assert len(r["images"]) == len(r["image_roles"]) == r["num_images"]
        assert tuple(r["image_roles"]) in LAYOUTS
        assert r["stage"] in ("low", "mid", "high")


def test_images_exist():
    missing = [p for r in load_rows() for p in r["images"] if not (DATA_ROOT / p).is_file()]
    assert not missing, missing[:5]


def test_task_order():
    assert task_names() == OLD_ORDER


def test_task_loader():
    loader = TaskLoader("left_embedded_figures")
    info = loader.get_task_info()
    assert info == {"stage": "mid", "process": "figure_judgement", "task": "left_embedded_figures",
                    "task_type": "match_to_sample", "num_images": 4}
    trial = loader.get_trials()[0]
    assert trial["trial_id"] == "trial_000" and trial["answer_key"] == "3"
    assert trial["data_root"] == DATA_ROOT
