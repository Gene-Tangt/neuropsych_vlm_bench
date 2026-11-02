"""Flatten the per-task test_specs/*_meta.json files into JSONL, one trial per line.

Each line is a self-contained record:

    {"index": 0, "id": "borb_orientation/trial_001",
     "stage": "low", "process": "simple_element_judgements", "task": "borb_orientation",
     "task_type": "yes_no", "num_stim": "one",
     "prompt": "This is a task. ...",
     "images": ["datasets/low/borb_orientation/001.png"],
     "image_roles": ["target"],
     "answer": "yes"}

`images` is in presentation order (the order ModelRunner.prepare_images sends them) and
`image_roles` names the role of each image in the original `images` dict, so the
original trial can be rebuilt exactly (see `to_trial`, used by --verify).

Usage:
    python scripts/json_to_jsonl.py                                  # -> test_specs/neuropsych_bench.jsonl
    python scripts/json_to_jsonl.py --out bench.jsonl --per-task test_specs/jsonl
    python scripts/json_to_jsonl.py --verify --check-images
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_FIELDS = ["stage", "process", "task", "task_type", "num_stim"]


def ordered_images(images: dict) -> tuple[list, list]:
    """Image paths in presentation order and the role of each one."""
    roles = []
    if "options" in images:
        roles += ["options"] * len(images["options"])
    if "target" in images:
        roles += ["target"] * len(images["target"])
    i = 1
    while f"option_{i}" in images:
        roles += [f"option_{i}"] * len(images[f"option_{i}"])
        i += 1
    expected = sum(len(v) for v in images.values())
    if len(roles) != expected:
        raise ValueError(f"unrecognised image roles: {sorted(images)}")
    # Pop each role's images in order so repeated roles (e.g. 'options') keep their order
    queues = {role: list(paths) for role, paths in images.items()}
    paths = [queues[role].pop(0).replace("\\", "/") for role in roles]
    return paths, roles


def to_records(task_path: Path) -> list:
    """One flat record per trial of a *_meta.json file."""
    with open(task_path, encoding="utf-8") as f:
        task = json.load(f)
    records = []
    for trial in task["trials"]:
        paths, roles = ordered_images(trial["images"])
        records.append({
            "id": f"{task['task']}/{trial['trial']}",
            **{key: task[key] for key in TASK_FIELDS},
            "prompt": task["prompt"],
            "images": paths,
            "image_roles": roles,
            "answer": trial["answer_key"],
        })
    return records


def to_trial(record: dict) -> dict:
    """Rebuild the original trial dict from a record."""
    images = {}
    for role, path in zip(record["image_roles"], record["images"]):
        images.setdefault(role, []).append(path)
    return {"trial": record["id"].split("/", 1)[1], "images": images, "answer_key": record["answer"]}


def verify(task_path: Path, records: list) -> None:
    """Raise if the records do not rebuild the original task file."""
    with open(task_path, encoding="utf-8") as f:
        task = json.load(f)
    rebuilt = [to_trial(r) for r in records]
    original = [{**t, "images": {k: [p.replace("\\", "/") for p in v] for k, v in t["images"].items()}}
                for t in task["trials"]]
    if rebuilt != original:
        raise AssertionError(f"round trip mismatch in {task_path}")
    for r in records:
        if any(r[key] != task[key] for key in TASK_FIELDS + ["prompt"]):
            raise AssertionError(f"task field mismatch in {task_path}")


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test-list", default="test_specs/test_list.json",
                        help="stage/task list giving which files to convert and in what order")
    parser.add_argument("--out", default="test_specs/neuropsych_bench.jsonl", help="combined JSONL output")
    parser.add_argument("--per-task", metavar="DIR", help="also write DIR/<stage>/<task>.jsonl per task")
    parser.add_argument("--verify", action="store_true", help="check every record rebuilds its source trial")
    parser.add_argument("--check-images", action="store_true", help="report image paths missing on disk")
    args = parser.parse_args()

    with open(ROOT / args.test_list, encoding="utf-8") as f:
        task_files = [ROOT / p for stage in json.load(f) for p in stage["task_paths"]]

    all_records, missing = [], []
    for task_path in task_files:
        records = to_records(task_path)
        if args.verify:
            verify(task_path, records)
        if args.check_images:
            missing += [p for r in records for p in r["images"] if not (ROOT / p).exists()]
        if args.per_task:
            write_jsonl(ROOT / args.per_task / records[0]["stage"] / f"{records[0]['task']}.jsonl", records)
        all_records += records

    ids = [r["id"] for r in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record ids")
    for i, r in enumerate(all_records):  # global integer index, as VLMEvalKit TSVs expect
        all_records[i] = {"index": i, **r}
    write_jsonl(ROOT / args.out, all_records)

    print(f"{len(all_records)} trials from {len(task_files)} tasks -> {args.out}")
    if args.verify:
        print("verify: every record round-trips to its source trial")
    if args.check_images:
        print(f"check-images: {len(missing)} missing" + (f", e.g. {missing[:3]}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
