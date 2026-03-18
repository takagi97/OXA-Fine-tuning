#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Find all files named generation_config.json in the given path
- Overwrite their content with the specified JSON
- Create .bak backups by default; can be disabled with --no-backup
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

TARGET_FILENAME = "generation_config.json"
TARGET_CONTENT = {
    "bos_token_id": 151643,
    "eos_token_id": [151643, 151645],
    "pad_token_id": 151643,
    "transformers_version": "4.51.0",
}

def overwrite_file(path: Path, make_backup: bool = True) -> bool:
    try:
        if make_backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
        with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
            json.dump(TARGET_CONTENT, tf, ensure_ascii=False, indent=2)
            tf.write("\n")
            tmp_name = tf.name
        os.replace(tmp_name, path)
        return True
    except Exception as e:
        print(f"[Error] Write failed:{path} -> {e}", file=sys.stderr)
        return False

def find_targets(root: Path):
    if root.is_file():
        return [root] if root.name == TARGET_FILENAME else []
    results = []
    for p in root.rglob(TARGET_FILENAME):
        if p.is_file():
            results.append(p)
    return results

def already_correct(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data == TARGET_CONTENT
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Overwrite generation_config.json in the given path with specified content"
    )
    parser.add_argument(
        "path",
        help="The file or directory path to process (files must be named generation_config.json)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not generate .bak backup files"
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"[Error] Path does not exist:{root}", file=sys.stderr)
        sys.exit(1)

    targets = find_targets(root)
    if not targets:
        print(f"[Note] No files found for {TARGET_FILENAME} files.")
        sys.exit(0)

    total = len(targets)
    changed = 0
    skipped = 0

    for t in targets:
        if already_correct(t):
            print(f"[Skip] Already matches target content: {t}")
            skipped += 1
            continue
        ok = overwrite_file(t, make_backup=not args.no_backup)
        if ok:
            print(f"[Done] Overwritten: {t}")
            changed += 1

    print(f"\nStats: Total found {total} files, overwritten {changed}, skipped {skipped}.")

if __name__ == "__main__":
    main()
