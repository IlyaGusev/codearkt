import uuid
import json
import socket
import random
import shutil
from contextlib import suppress
from typing import Optional, Dict, Any, List
from pathlib import Path

MAX_LENGTH_TRUNCATE_CONTENT: int = 20000


def get_unique_id() -> str:
    return str(uuid.uuid4())[:8]


def find_free_port() -> Optional[int]:
    ports = list(range(5000, 6001))
    random.shuffle(ports)
    for port in ports:
        try:
            with socket.socket() as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return None


def truncate_content(
    content: str,
    max_length: int = MAX_LENGTH_TRUNCATE_CONTENT,
    prefix_only: bool = False,
    suffix_only: bool = False,
    target_line: Optional[int] = None,
) -> str:
    assert int(prefix_only) + int(suffix_only) + int(target_line is not None) <= 1
    disclaimer = (
        f"\n\n..._This content has been truncated to stay below {max_length} characters_...\n\n"
    )
    half_length = max_length // 2
    if len(content) <= max_length:
        return content

    if prefix_only:
        prefix = content[:max_length]
        return prefix + disclaimer

    elif suffix_only:
        suffix = content[-max_length:]
        return disclaimer + suffix

    elif target_line:
        line_start_pos = 0
        next_pos = content.find("\n") + 1
        line_end_pos = next_pos
        for _ in range(target_line):
            next_pos = content.find("\n", next_pos) + 1
            line_start_pos = line_end_pos
            line_end_pos = next_pos
        assert line_end_pos >= line_start_pos
        length = line_end_pos - line_start_pos
        remaining_length = max(0, max_length - length)
        half_length = remaining_length // 2
        start = max(0, line_start_pos - half_length)
        end = min(len(content), line_end_pos + half_length)
        final_content = content[start:end]
        if start == 0:
            return final_content + disclaimer
        elif end == len(content):
            return disclaimer + final_content
        return disclaimer + content[start:end] + disclaimer

    prefix = content[:half_length]
    suffix = content[-half_length:]
    return prefix + disclaimer + suffix


def is_correct_json(content: str) -> bool:
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        return False


def append_jsonl_atomic(output_path: Path, record: Dict[str, Any]) -> None:
    directory = output_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        # Best-effort file lock (Linux). Safe to no-op if fcntl not available
        with suppress(Exception):
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        existing: List[Dict[str, Any]] = []
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = [json.loads(line) for line in f]
            except Exception:
                existing = []

        existing.append(record)

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in existing:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        shutil.move(tmp_path, output_path)

        with suppress(Exception):
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
