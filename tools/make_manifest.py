# -*- coding: utf-8 -*-
"""Манифест поставки: путь, размер и SHA-256 каждого файла, который уходит в релиз.

    python tools/make_manifest.py

Хеши пишутся машиной, а не рукой: вписанный вручную хеш расходится с файлом после первой
же правки, и заметить это некому. Список файлов берётся у git (отслеживаемые плюс новые,
не попавшие под .gitignore), сам манифест в него не входит — он не может хранить свой хеш.
"""
import os
import json
import hashlib
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "4.3.0"
TD_BUILD = "2025.32460"
NAME = "CANDIDATE_MANIFEST.json"


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def main():
    listed = git("-c", "core.quotepath=off", "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    files = []
    for rel in sorted(set(listed)):
        full = os.path.join(ROOT, rel)
        if rel == NAME or not os.path.isfile(full):
            continue
        with open(full, "rb") as f:
            data = f.read()
        files.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    accept = {}
    try:
        with open(os.path.join(ROOT, "native_acceptance.json"), encoding="utf-8") as f:
            a = json.load(f)
        accept = {"passed": a["passed"], "total": a["total"], "date": a["date"],
                  "tox": a["tox"], "tox_sha256": a["tox_sha256"]}
    except OSError:
        pass

    tox = "WEDOAUDIO_%s_TD%s.tox" % (VERSION, TD_BUILD)
    on_disk = next((x["sha256"] for x in files if x["path"] == tox), None)
    doc = {
        "version": VERSION,
        "published": False,
        "tdBuild": TD_BUILD,
        "component": tox,
        "channels": {"base": 22, "loudness": 6},
        "upstreamBase": git("merge-base", "HEAD", "origin/master").strip(),
        "nativeAcceptance": accept,
        # приёмка обязана относиться к тому самому файлу, который лежит в поставке
        "acceptanceMatchesComponent": bool(on_disk) and on_disk == accept.get("tox_sha256"),
        "files": files,
    }
    with open(os.path.join(ROOT, NAME), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("%s: %d files, acceptance matches component: %s" % (NAME, len(files), doc["acceptanceMatchesComponent"]))


if __name__ == "__main__":
    main()
