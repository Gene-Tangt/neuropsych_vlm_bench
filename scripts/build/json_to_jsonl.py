"""Flatten the old per-task test_specs/*_meta.json files into data/neuropsych_bench.jsonl, one trial per line.

This is how data/neuropsych_bench.jsonl was produced. test_specs/ was deleted once the JSONL
replaced it, so the script only runs against the git history from before the restructure;
it is kept as the record of the conversion. Edit the JSONL directly from now on.

Each line is a self-contained record:

    {"index": 0, "id": "borb_orientation/trial_001",
     "stage": "low", "process": "simple_element_judgements", "task": "borb_orientation",
     "task_type": "yes_no", "num_images": 1,
     "prompt": "This is a task. ...",
     "images": ["images/low/borb_orientation/001.png"],
     "image_roles": ["target"],
     "answer": "yes"}

`images` is in presentation order and relative to the JSONL's folder (data/); the images
themselves were moved from datasets/ to data/images/. `image_roles` names the role of each
image in the original `images` dict, so the original trial can be rebuilt exactly (see
`to_trial`, used by --verify). `num_images` replaces the old `num_stim` word ("one".."four").

Usage (from the pre-restructure tree):
    python scripts/build/json_to_jsonl.py --verify --check-images
    python scripts/build/json_to_jsonl.py --per-task data/per_task
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_PREFIX, NEW_PREFIX = "datasets/", "images/"  # old repo-relative image folder -> path relative to data/
TASK_FIELDS = ["stage", "process", "task", "task_type"]


def new_path(path: str) -> str:
    path = path.replace("\\", "/")
    if not path.startswith(OLD_PREFIX):
        raise ValueError(f"image path outside {OLD_PREFIX}: {path}")
    return NEW_PREFIX + path[len(OLD_PREFIX):]


def ordered_images(images: dict) -> tuple:
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
    if len(roles) != sum(len(v) for v in images.values()):
        raise ValueError(f"unrecognised image roles: {sorted(images)}")
    # Pop each role's images in order so repeated roles (e.g. 'options') keep their order
    queues = {role: list(paths) for role, paths in images.items()}
    paths = [new_path(queues[role].pop(0)) for role in roles]
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
            "num_images": len(paths),
            "prompt": task["prompt"],
            "images": paths,
            "image_roles": roles,
            "answer": trial["answer_key"],
        })
    return records


def to_trial(record: dict) -> dict:
    """Rebuild the original trial dict (with old datasets/ paths) from a record."""
    images = {}
    for role, path in zip(record["image_roles"], record["images"]):
        images.setdefault(role, []).append(OLD_PREFIX + path[len(NEW_PREFIX):])
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
    parser.add_argument("--out", default="data/neuropsych_bench.jsonl", help="combined JSONL output")
    parser.add_argument("--per-task", metavar="DIR", help="also write DIR/<stage>/<task>.jsonl per task")
    parser.add_argument("--verify", action="store_true", help="check every record rebuilds its source trial")
    parser.add_argument("--check-images", action="store_true",
                        help="report images missing from the old datasets/ folder or the new data/images/ folder")
    args = parser.parse_args()

    with open(ROOT / args.test_list, encoding="utf-8") as f:
        task_files = [ROOT / p for stage in json.load(f) for p in stage["task_paths"]]

    out = ROOT / args.out
    all_records, missing = [], []
    for task_path in task_files:
        records = to_records(task_path)
        if args.verify:
            verify(task_path, records)
        if args.check_images:
            missing += [p for r in records for p in r["images"]
                        if not (out.parent / p).exists() and not (ROOT / OLD_PREFIX / p[len(NEW_PREFIX):]).exists()]
        if args.per_task:
            write_jsonl(ROOT / args.per_task / records[0]["stage"] / f"{records[0]['task']}.jsonl", records)
        all_records += records

    ids = [r["id"] for r in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record ids")
    all_records = [{"index": i, **r} for i, r in enumerate(all_records)]  # global integer index, as VLMEvalKit expects
    write_jsonl(out, all_records)

    print(f"{len(all_records)} trials from {len(task_files)} tasks -> {args.out}")
    if args.verify:
        print("verify: every record round-trips to its source trial")
    if args.check_images:
        print(f"check-images: {len(missing)} missing" + (f", e.g. {missing[:3]}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
