#!/usr/bin/env python3
"""ImputeForge — three-tier degradation GitHub delivery helper.

Usage:
    python scripts/gh_push.py --repo imputeforge --local . \
        --tag v0.1.0 --description "..." --release "<honest notes>"

Tiers (graceful degradation, never silent-fail):
    T1  create repo (best-effort) -> commit -> tag -> push --follow-tags -> gh release create
    T2  if release create fails: keep commit+tag+push, skip release
    T3  if push fails: keep local commit+tag, print manual-push instructions

Design rules followed:
    * author署名 晨星; core.autocrlf=false (Windows safe)
    * 绝不把 A 级成果标为 S; release 正文须含诚实门禁结论
    * 零硬编码密钥: 全部走 gh keyring / git config
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

AUTHOR_NAME = "晨星"
AUTHOR_EMAIL = "CJX0712@users.noreply.github.com"
GH_USER = "CJX0712"


def run(cmd, check=True, cwd=None):
    print(f"$ {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as e:
        print(f"  [ERR] command not found: {e}", file=sys.stderr)
        return None
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        for line in out.strip().splitlines()[:20]:
            print("  |", line)
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p


def branch_name(cwd: Path) -> str:
    p = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    return (p.stdout or "master").strip() or "master"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="remote repo name (under GH_USER)")
    ap.add_argument("--local", default=".", help="local project path")
    ap.add_argument("--tag", required=True, help="git tag, e.g. v0.1.0")
    ap.add_argument("--description", default="", help="repo description")
    ap.add_argument("--release", default="", help="release notes (markdown)")
    args = ap.parse_args()

    cwd = Path(args.local).resolve()
    remote = f"https://github.com/{GH_USER}/{args.repo}.git"

    # 0. git hygiene
    run(["git", "config", "user.name", AUTHOR_NAME], cwd=cwd)
    run(["git", "config", "user.email", AUTHOR_EMAIL], cwd=cwd)
    run(["git", "config", "core.autocrlf", "false"], cwd=cwd)

    # 1. best-effort repo create (ignore if exists)
    try:
        run(
            [
                "gh",
                "repo",
                "create",
                f"{GH_USER}/{args.repo}",
                "--public",
                "--description",
                args.description or args.repo,
            ],
            cwd=cwd,
        )
    except subprocess.CalledProcessError:
        print("  [info] repo already exists or create skipped; continuing.")

    # 2. commit
    run(["git", "add", "-A"], cwd=cwd)
    st = run(["git", "status", "--porcelain"], cwd=cwd)
    if st.stdout.strip():
        msg = f"{args.repo} v0.1.0 — 晨星"
        run(["git", "commit", "-m", msg], cwd=cwd)
    else:
        print("  [info] nothing to commit.")

    # 3. tag
    try:
        run(["git", "tag", "-f", args.tag], cwd=cwd)
    except subprocess.CalledProcessError:
        print("  [info] tag exists; forcing.")

    br = branch_name(cwd)
    # 4. push (T1/T2 share this; T3 falls back here on failure)
    try:
        run(["git", "push", "-u", "origin", br, "--follow-tags"], cwd=cwd)
        # set upstream alias if absent
        run(["git", "remote", "add", "origin", remote], cwd=cwd, check=False)
        run(["git", "push", "-u", "origin", br, "--follow-tags"], cwd=cwd)
    except subprocess.CalledProcessError:
        # T3: push failed — keep local, instruct
        print("\n[T3] PUSH FAILED. Local commit+tag preserved. Manual:")
        print(f"    git remote add origin {remote}")
        print(f"    git push -u origin {br} --follow-tags")
        return 3

    # 5. release (T1); on failure -> T2 (skip release, still delivered)
    try:
        run(
            [
                "gh",
                "release",
                "create",
                args.tag,
                "--title",
                f"{args.repo} {args.tag}",
                "--notes",
                args.release or f"{args.repo} {args.tag} (see README).",
            ],
            cwd=cwd,
        )
        print(f"\n[T1] DELIVERED + RELEASED: {args.tag}")
        return 0
    except subprocess.CalledProcessError:
        print(
            f"\n[T2] PUSHED (tag {args.tag}) but release create failed; "
            f"deliverable intact on GitHub."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
