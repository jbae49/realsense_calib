import json
from pathlib import Path


def parse_tag_size_map(raw):
    mapping = {}
    if not raw:
        return mapping
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid tag-size map entry: {token}")
        tag_id_str, size_str = token.split(":", 1)
        mapping[int(tag_id_str.strip())] = float(size_str.strip())
    return mapping


def load_tag_size_config(path):
    p = Path(path)
    if not p.exists():
        return None, {}
    data = json.loads(p.read_text())
    default_size = float(data.get("default_tag_size_m", 0.077))
    raw_map = data.get("per_tag_size_m", {})
    tag_size_map = {int(k): float(v) for k, v in raw_map.items()}
    return default_size, tag_size_map


def merge_tag_sizes(default_size, base_map, override_map):
    merged = dict(base_map)
    merged.update(override_map)
    return default_size, merged


def detect_with_tag_sizes(detector, gray, camera_params, default_tag_size, tag_size_map):
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=default_tag_size,
    )
    by_id = {det.tag_id: det for det in detections}

    if tag_size_map:
        for custom_size in sorted(set(tag_size_map.values())):
            mapped_ids = {tag_id for tag_id, sz in tag_size_map.items() if sz == custom_size}
            custom_dets = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=custom_size,
            )
            for det in custom_dets:
                if det.tag_id in mapped_ids:
                    by_id[det.tag_id] = det

    return [by_id[k] for k in sorted(by_id.keys())]
