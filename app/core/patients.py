"""
core/patients.py — Read patient history and individual POD records from JSON sidecars.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_pid_pod(filename: str):
    pid = re.search(r"(PT\d{4})", filename)
    pod = re.search(r"_POD(\d+)", filename)
    return (pid.group(1) if pid else None), (int(pod.group(1)) if pod else None)


def get_patient_history(dataset_root: str, patient_id: str) -> List[Dict[str, Any]]:
    """
    Return all daily observation dicts for a given patient_id,
    sorted by POD ascending.
    """
    json_dir = Path(dataset_root).resolve() / "jsons"
    if not json_dir.exists():
        raise FileNotFoundError(f"jsons directory not found: {json_dir}")

    observations = []
    for jp in sorted(json_dir.glob("*.json")):
        pid, pod = _parse_pid_pod(jp.name)
        if pid != patient_id:
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            dob = data.get("daily_observation", {})
            observations.append(dob)
        except Exception:
            continue

    observations.sort(key=lambda x: x.get("pod", 0))
    return observations


def get_patient_pod_record(
    dataset_root: str, patient_id: str, pod: int
) -> Optional[Dict[str, Any]]:
    """
    Return the full JSON sidecar content for a specific patient + POD,
    or None if not found.
    """
    json_dir = Path(dataset_root).resolve() / "jsons"
    if not json_dir.exists():
        raise FileNotFoundError(f"jsons directory not found: {json_dir}")

    for jp in json_dir.glob("*.json"):
        pid, p = _parse_pid_pod(jp.name)
        if pid == patient_id and p == pod:
            try:
                return json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def list_all_patients(dataset_root: str) -> List[str]:
    """Return a sorted list of unique patient IDs found in jsons/."""
    json_dir = Path(dataset_root).resolve() / "jsons"
    if not json_dir.exists():
        return []
    ids = set()
    for jp in json_dir.glob("*.json"):
        pid, _ = _parse_pid_pod(jp.name)
        if pid:
            ids.add(pid)
    return sorted(ids)