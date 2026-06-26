import os
import pickle as pkl
import re
from pathlib import Path

import numpy as np

from utils.darpa_parser import metadata as darpa_metadata


_EXEC_RE = re.compile(r'"exec":"(.*?)"')
_SUBJECT_RE = re.compile(r'subject":\{"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID":"(.*?)"\}')
_PREDICATE_RE = re.compile(r'predicateObject":\{"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID":"(.*?)"\}')
_PREDICATE2_RE = re.compile(r'predicateObject2":\{"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID":"(.*?)"\}')


def _file_signature(paths):
    signature = []
    for path in paths:
        stat = os.stat(path)
        signature.append((str(path), int(stat.st_size), int(stat.st_mtime)))
    return signature


def _extract_exec(line):
    values = _EXEC_RE.findall(line)
    if len(values) == 0:
        return None
    return values[-1]


def _iter_event_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "com.bbn.tc.schema.avro.cdm18.Event" in line:
                yield line


def _collect_train_execs(data_dir, train_files):
    train_execs = set()
    for file_name in train_files:
        for line in _iter_event_lines(data_dir / file_name):
            if '"exec"' not in line:
                continue
            exec_name = _extract_exec(line)
            if exec_name is not None:
                train_execs.add(exec_name)
    return train_execs


def _collect_unseen_exec_entities(data_dir, test_files, train_execs):
    entities = set()
    unseen_execs = {}
    uuid_patterns = (_SUBJECT_RE, _PREDICATE_RE, _PREDICATE2_RE)
    for file_name in test_files:
        for line in _iter_event_lines(data_dir / file_name):
            if '"exec"' not in line:
                continue
            exec_name = _extract_exec(line)
            if exec_name is None or exec_name in train_execs:
                continue
            touched = []
            for pattern in uuid_patterns:
                values = pattern.findall(line)
                if len(values) > 0 and values[0] != "null":
                    touched.append(values[0])
            if len(touched) == 0:
                continue
            bucket = unseen_execs.setdefault(exec_name, set())
            bucket.update(touched)
            entities.update(touched)
    return entities, unseen_execs


def _build_test_node_map_from_txt(data_dir, test_files):
    """Rebuild the UUID -> concatenated test node id map used by darpa_parser."""
    offset = 0
    result = {}
    for file_name in test_files:
        rows = []
        with open(data_dir / "{}.txt".format(file_name), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 6:
                    continue
                src, _src_type, dst, _dst_type, edge_type, ts = parts
                if "READ" in edge_type or "RECV" in edge_type or "LOAD" in edge_type:
                    rows.append((dst, src, int(ts)))
                else:
                    rows.append((src, dst, int(ts)))

        rows.sort(key=lambda row: row[2])
        local_map = {}
        for src, dst, _ts in rows:
            if src not in local_map:
                local_map[src] = len(local_map)
            if dst not in local_map:
                local_map[dst] = len(local_map)
        for uuid, local_id in local_map.items():
            result[uuid] = offset + local_id
        offset += len(local_map)
    return result


def load_or_build_cadets_exec_context_score(expected_len, cache_dir="./result"):
    """Return a binary score for CADETS nodes touched by test-time unseen exec values.

    This is unsupervised: it only learns the set of exec names present in the
    training raw logs, then marks test entities touched by exec names not seen
    during training. CADETS attack FileObjects have no path/name/hash in CDM, so
    this preserves event context that the original type-only preprocessing drops.
    """
    data_dir = Path("data") / "cadets"
    split = darpa_metadata.get("cadets")
    if split is None:
        return None, {"reason": "missing cadets split metadata"}

    train_files = split["train"]
    test_files = split["test"]
    raw_paths = [data_dir / name for name in train_files + test_files]
    txt_paths = [data_dir / "{}.txt".format(name) for name in test_files]
    required_paths = raw_paths + txt_paths
    missing = [str(path) for path in required_paths if not path.exists()]
    if len(missing) > 0:
        return None, {"reason": "missing raw/txt files", "missing": missing[:5]}

    signature = _file_signature(required_paths)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir) / "cadets_exec_context_score.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pkl.load(f)
        if (
            cached.get("expected_len") == expected_len
            and cached.get("signature") == signature
            and isinstance(cached.get("score"), np.ndarray)
            and cached["score"].shape[0] == expected_len
        ):
            details = dict(cached.get("details", {}))
            details["cache"] = str(cache_path)
            details["cache_hit"] = True
            return cached["score"], details

    train_execs = _collect_train_execs(data_dir, train_files)
    entities, unseen_execs = _collect_unseen_exec_entities(data_dir, test_files, train_execs)
    node_map = _build_test_node_map_from_txt(data_dir, test_files)

    score = np.zeros(expected_len, dtype=np.float32)
    mapped = 0
    for uuid in entities:
        node_id = node_map.get(uuid)
        if node_id is None or node_id >= expected_len:
            continue
        score[node_id] = 1.0
        mapped += 1

    top_unseen = sorted(
        ((name, len(values)) for name, values in unseen_execs.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    details = {
        "train_exec_count": len(train_execs),
        "unseen_exec_count": len(unseen_execs),
        "candidate_uuid_count": len(entities),
        "mapped_node_count": mapped,
        "positive_score_count": int(score.sum()),
        "top_unseen_execs": top_unseen,
        "cache": str(cache_path),
        "cache_hit": False,
    }
    with open(cache_path, "wb") as f:
        pkl.dump(
            {
                "expected_len": expected_len,
                "signature": signature,
                "score": score,
                "details": details,
            },
            f,
        )
    return score, details
