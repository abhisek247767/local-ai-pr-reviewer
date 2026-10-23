#!/usr/bin/env python3
"""SlothReview: a lazy, local AI reviewer for GitHub pull requests.

Polls the repos listed in .env, reviews new or updated PRs with a local
Ollama model, and posts one COMMENT review per PR head commit.
It never approves, requests changes, pushes, or commits.
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from local_ai_pr_reviewer.guardrails import (
    FILE_REVIEW_SCHEMA,
    OVERVIEW_SCHEMA,
    SEVERITIES,
    check_file_review,
    check_overview,
)
from local_ai_pr_reviewer.promts.promt import (
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_summary_prompt,
    build_user_prompt,
)

GITHUB_API = "https://api.github.com"
MARKER = "<!-- slothreview:{sha} -->"
MAX_REVIEW_BODY = 60_000  # GitHub's limit is 65,536 characters
SEVERITY_ICONS = {"critical": "🔴", "suggestion": "🟡", "nit": "⚪"}

DEFAULT_SKIP_PATTERNS = [
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "go.sum", "*.min.js", "*.min.css", "*.map", "*.snap",
    "dist/*", "build/*", "vendor/*", "node_modules/*",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg", "*.pdf",
    "*.woff", "*.woff2", "*.ttf", "*.zip",
]

FENCE_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript",
    ".tsx": "tsx", ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".css": "css", ".scss": "scss",
    ".html": "html", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".sql": "sql", ".sh": "bash", ".kt": "kotlin", ".swift": "swift", ".php": "php",
}

log = logging.getLogger("slothreview")



@dataclass
class Config:
    token: str
    repos: list
    model: str
    ollama_url: str
    num_ctx: int
    poll_interval: int
    state_file: Path
    max_patch_chars: int
    max_files: int
    skip_drafts: bool
    dry_run: bool
    skip_patterns: list


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        sys.exit(f"{name} must be a whole number, got {value!r}")


def load_config() -> Config:
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN is missing. Copy .env.example to .env and fill it in.")

    repos = [r.strip() for r in os.getenv("REPOS", "").split(",") if r.strip()]
    if not repos:
        sys.exit("REPOS is missing. Example: REPOS=yourname/repo-one,yourname/repo-two")
    for repo in repos:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            sys.exit(f"Bad repo name {repo!r}. Use the owner/repo format.")

    extra_skips = [p.strip() for p in os.getenv("SKIP_PATTERNS", "").split(",") if p.strip()]

    return Config(
        token=token,
        repos=repos,
        model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b").strip(),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/"),
        num_ctx=env_int("OLLAMA_NUM_CTX", 8192),
        poll_interval=env_int("POLL_INTERVAL_SECONDS", 300),
        state_file=Path(os.getenv("STATE_FILE", "state.json").strip()),
        max_patch_chars=env_int("MAX_PATCH_CHARS", 12_000),
        max_files=env_int("MAX_FILES_PER_PR", 25),
        skip_drafts=env_bool("SKIP_DRAFTS", True),
        dry_run=env_bool("DRY_RUN", False),
        skip_patterns=DEFAULT_SKIP_PATTERNS + extra_skips,
    )



def load_state(path: Path) -> dict:
    """state.json maps "owner/repo#123" to the last reviewed head commit SHA."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s); starting fresh.", path, exc)
        return {}


def save_state(path: Path, state: dict) -> None:
    """Write atomically so a crash mid-write never corrupts the file."""
    directory = path.resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# --------------------------------------------------------------------------- github

class GitHub:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "slothreview",
        })

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        if not url.startswith("http"):
            url = GITHUB_API + url
        response = None
        for attempt in range(4):
            response = self.session.request(method, url, timeout=30, **kwargs)
            if response.status_code in (403, 429) and response.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = min(max(reset - int(time.time()), 1) + 1, 900)
                log.warning("GitHub rate limit hit; sleeping %ss.", wait)
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response
        response.raise_for_status()
        return response

    def _paginate(self, path: str, params: dict | None = None, limit: int = 1000) -> list:
        url = GITHUB_API + path
        params = dict(params or {}, per_page=100)
        items: list = []
        while url and len(items) < limit:
            response = self._request("GET", url, params=params)
            items.extend(response.json())
            url = response.links.get("next", {}).get("url")
            params = None  # the "next" URL already carries the query string
        return items[:limit]

    def open_pulls(self, repo: str) -> list:
        return self._paginate(f"/repos/{repo}/pulls", {"state": "open"})

    def pr_files(self, repo: str, number: int) -> list:
        return self._paginate(f"/repos/{repo}/pulls/{number}/files", limit=3000)

    def reviews(self, repo: str, number: int) -> list:
        return self._paginate(f"/repos/{repo}/pulls/{number}/reviews")

    def post_comment_review(self, repo: str, number: int, sha: str, body: str, comments: list) -> None:
        # The event is always COMMENT. This tool never approves or blocks a PR.
        payload = {"commit_id": sha, "body": body, "event": "COMMENT"}
        if comments:
            payload["comments"] = comments
        self._request("POST", f"/repos/{repo}/pulls/{number}/reviews", json=payload)



def parse_model_json(text: str) -> dict:
    text = text.strip()
    candidates = [text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text)]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    log.warning("Model did not return valid JSON; keeping raw text as the summary.")
    return {"summary": text[:500], "issues": []}


def ask_ollama(cfg: Config, system_prompt: str, user_prompt: str, schema: dict) -> dict:
    response = requests.post(
        f"{cfg.ollama_url}/api/chat",
        json={
            "model": cfg.model,
            "stream": False,
            "format": schema,  # structured output: Ollama constrains the reply to this JSON schema
            "options": {"temperature": 0.2, "num_ctx": cfg.num_ctx},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=900,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return parse_model_json(content)


def check_ollama(cfg: Config) -> None:
    try:
        response = requests.get(f"{cfg.ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        sys.exit(f"Cannot reach Ollama at {cfg.ollama_url} ({exc}). Is `ollama serve` running?")
    names = {m.get("name", "") for m in response.json().get("models", [])}
    if cfg.model not in names and f"{cfg.model}:latest" not in names:
        log.warning("Model %r not found locally. Run: ollama pull %s", cfg.model, cfg.model)


# --------------------------------------------------------------------------- review

def is_skipped(filename: str, patterns: list) -> bool:
    base = filename.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(filename, p) or fnmatch.fnmatch(base, p) for p in patterns)


def select_files(cfg: Config, files: list):
    reviewable, skipped = [], []
    for f in files:
        name = f.get("filename", "?")
        if f.get("status") == "removed":
            skipped.append((name, "deleted"))
        elif is_skipped(name, cfg.skip_patterns):
            skipped.append((name, "matches skip pattern"))
        elif not f.get("patch"):
            skipped.append((name, "binary or diff too large for GitHub"))
        elif len(reviewable) >= cfg.max_files:
            skipped.append((name, f"over the {cfg.max_files}-file limit"))
        else:
            reviewable.append(f)
    return reviewable, skipped


def review_file(cfg: Config, pr: dict, f: dict) -> dict:
    patch = f["patch"]
    truncated = len(patch) > cfg.max_patch_chars
    prompt = build_user_prompt(
        pr_title=pr.get("title", ""),
        pr_body=pr.get("body") or "",
        filename=f["filename"],
        status=f.get("status", "modified"),
        patch=patch[: cfg.max_patch_chars],
        truncated=truncated,
    )
    data = ask_ollama(cfg, SYSTEM_PROMPT, prompt, FILE_REVIEW_SCHEMA)
    result = check_file_review(data, patch[: cfg.max_patch_chars], f["filename"])
    result["truncated"] = truncated
    return result


def summarize_pr(cfg: Config, pr: dict, results: list, all_issues: list) -> str:
    """One extra model call that writes the PR-level overview from the per-file results."""
    file_notes = [(f["filename"], r["summary"]) for f, r in results if r["summary"]]
    prompt = build_summary_prompt(pr.get("title", ""), file_notes, all_issues)
    try:
        overview = check_overview(ask_ollama(cfg, SUMMARY_SYSTEM_PROMPT, prompt, OVERVIEW_SCHEMA))
    except requests.RequestException as exc:
        log.warning("  Summary call failed (%s); using per-file notes instead.", exc)
        overview = ""
    return overview or " ".join(summary for _, summary in file_notes)


def overall_verdict(counts: dict) -> str:
    # Text only: the review is still posted as a COMMENT and never blocks the PR.
    if counts["critical"]:
        return "Request Changes"
    if counts["suggestion"] or counts["nit"]:
        return "Comment"
    return "Looks Good"


def issue_label(issue: dict) -> str:
    return issue["title"] or textwrap.shorten(issue["problem"].split(". ")[0], 90, placeholder="…")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def location(name: str, issue: dict) -> str:
    if not issue["line"]:
        return f"`{name}`"
    if issue["start_line"] and issue["start_line"] != issue["line"]:
        return f"`{name}:{issue['start_line']}-{issue['line']}`"
    return f"`{name}:{issue['line']}`"


def issue_details(name: str, issue: dict) -> list:
    """Problem / Current code / Suggested fix, shared by inline comments and the review body."""
    lang = FENCE_LANG.get(Path(name).suffix.lower(), "")
    lines = [f"**Problem:** {issue['problem']}"]
    if issue["current_code"]:
        lines += ["", "**Current code:**", f"````{lang}", issue["current_code"], "````"]
    if issue["suggestion"] or issue["code"]:
        lines += ["", f"**Suggested fix:** {issue['suggestion']}".rstrip()]
        if issue["code"]:
            lines += [f"````{lang}", issue["code"], "````"]
    return lines


def inline_comment(name: str, issue: dict) -> dict:
    """One review comment attached to the exact changed lines in the diff."""
    body = "\n".join([f"**{SEVERITY_ICONS[issue['severity']]} {issue_label(issue)}**", ""] + issue_details(name, issue))
    comment = {"path": name, "line": issue["line"], "side": "RIGHT", "body": body}
    if issue["start_line"] and issue["start_line"] < issue["line"]:
        comment.update(start_line=issue["start_line"], start_side="RIGHT")
    return comment


def format_review(sha: str, overview: str, results: list, skipped: list, inline: bool = True):
    """Returns (review body, inline comments). Issues with a line in the diff become inline
    comments; the rest are written out in full in the body. inline=False puts everything in the body."""
    all_issues = [(f["filename"], issue) for f, result in results for issue in result["issues"]]
    order = {s: n for n, s in enumerate(SEVERITIES)}
    all_issues.sort(key=lambda pair: order[pair[1]["severity"]])
    counts = {s: sum(1 for _, i in all_issues if i["severity"] == s) for s in SEVERITIES}
    anchored = [(name, i) for name, i in all_issues if inline and i["line"]]
    in_body = [(name, i) for name, i in all_issues if not (inline and i["line"])]

    found = ", ".join(
        plural(counts[s], label)
        for s, label in (("critical", "critical"), ("suggestion", "suggestion"), ("nit", "nit"))
        if counts[s]
    ) or "none"
    lines = [
        "## PR Review Summary",
        f"**Overall:** {overall_verdict(counts)}",
        "",
        f"**Issues found:** {found}",
    ]
    if overview:
        lines += ["", overview]

    critical = [(name, i) for name, i in all_issues if i["severity"] == "critical"]
    if critical:
        lines += ["", "**Must fix before merge:**"]
        lines += [f"{n}. {issue_label(i)} ({location(name, i)})" for n, (name, i) in enumerate(critical, 1)]
    suggestions = [(name, i) for name, i in all_issues if i["severity"] != "critical"]
    if suggestions:
        lines += ["", "**Suggestions:**"]
        lines += [f"- {issue_label(i)} ({location(name, i)})" for name, i in suggestions]
    if anchored:
        lines += ["", f"_Details are in the {plural(len(anchored), 'inline comment')} on the changed lines._"]

    for n, (name, issue) in enumerate(in_body, 1):
        lines += [
            "", "---", "",
            f"### {SEVERITY_ICONS[issue['severity']]} {n}. {issue_label(issue)}",
            location(name, issue),
            "",
        ] + issue_details(name, issue)

    truncated = [f["filename"] for f, r in results if r.get("truncated")]
    if truncated:
        lines += ["", "_Diff truncated for: " + ", ".join(f"`{t}`" for t in truncated) + "_"]

    if skipped:
        lines += ["", "<details><summary>Skipped files</summary>", ""]
        lines += [f"- `{name}`: {reason}" for name, reason in skipped]
        lines += ["", "</details>"]

    marker = MARKER.format(sha=sha)
    body = "\n".join(lines)
    if len(body) > MAX_REVIEW_BODY:
        body = body[:MAX_REVIEW_BODY] + "\n\n_…review truncated to fit GitHub's size limit._"
    return f"{body}\n\n{marker}", [inline_comment(name, i) for name, i in anchored]


def process_pr(cfg: Config, gh: GitHub, repo: str, pr: dict, state: dict) -> None:
    number, sha = pr["number"], pr["head"]["sha"]
    key = f"{repo}#{number}"

    if state.get(key) == sha:
        return
    if cfg.skip_drafts and pr.get("draft"):
        log.info("%s is a draft; skipping for now.", key)
        return

    # Safety net if state.json was lost: never post twice for the same commit.
    if not cfg.dry_run:
        marker = MARKER.format(sha=sha)
        if any(marker in (r.get("body") or "") for r in gh.reviews(repo, number)):
            log.info("%s @ %s already reviewed; syncing state.", key, sha[:7])
            state[key] = sha
            return

    log.info("Reviewing %s @ %s: %s", key, sha[:7], pr.get("title", ""))
    reviewable, skipped = select_files(cfg, gh.pr_files(repo, number))

    results = []
    for f in reviewable:
        log.info("  · %s", f["filename"])
        try:
            results.append((f, review_file(cfg, pr, f)))
        except requests.RequestException as exc:
            log.error("  Model call failed for %s: %s", f["filename"], exc)
            skipped.append((f["filename"], "model error"))

    if reviewable and not results:
        log.error("Every model call failed for %s; will retry next cycle.", key)
        return

    all_issues = [(f["filename"], issue) for f, result in results for issue in result["issues"]]
    log.info("  · writing summary")
    overview = summarize_pr(cfg, pr, results, all_issues)
    body, comments = format_review(sha, overview, results, skipped)
    if cfg.dry_run:
        print(f"\n===== DRY RUN: {key} =====\n{body}\n")
        for c in comments:
            where = f"{c['path']}:{c.get('start_line', c['line'])}-{c['line']}"
            print(f"----- inline comment on {where} -----\n{c['body']}\n")
    else:
        try:
            gh.post_comment_review(repo, number, sha, body, comments)
        except requests.HTTPError as exc:
            # 422 = GitHub rejected a line anchor; post everything in the body instead.
            if not comments or exc.response is None or exc.response.status_code != 422:
                raise
            log.warning("  GitHub rejected the inline comments (%s); posting them in the summary.", exc)
            body, comments = format_review(sha, overview, results, skipped, inline=False)
            gh.post_comment_review(repo, number, sha, body, comments)
        log.info("Posted review on %s with %s inline comment(s).", key, len(comments))

    state[key] = sha


def run_once(cfg: Config, gh: GitHub) -> None:
    state = load_state(cfg.state_file)
    for repo in cfg.repos:
        try:
            pulls = gh.open_pulls(repo)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            hint = " Check the token has access to this repo." if code in (403, 404) else ""
            log.error("Could not list PRs for %s (HTTP %s).%s", repo, code, hint)
            continue

        # Forget closed or merged PRs so state.json stays small.
        open_keys = {f"{repo}#{p['number']}" for p in pulls}
        for key in [k for k in state if k.startswith(f"{repo}#") and k not in open_keys]:
            del state[key]

        for pr in pulls:
            try:
                process_pr(cfg, gh, repo, pr, state)
            except requests.RequestException as exc:
                reason = ""
                if getattr(exc, "response", None) is not None:
                    try:
                        reason = f" ({exc.response.json().get('message', '')})"
                    except ValueError:
                        pass
                    if exc.response.status_code == 403:
                        reason += " Token needs write access to pull requests on this repo."
                log.error("Failed on %s#%s: %s%s", repo, pr.get("number"), exc, reason)
            save_state(cfg.state_file, state)

    save_state(cfg.state_file, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="SlothReview: local AI PR reviewer")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit (good for cron)")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config()
    check_ollama(cfg)
    gh = GitHub(cfg.token)
    log.info("Watching %s with %s%s", ", ".join(cfg.repos), cfg.model, " (dry run)" if cfg.dry_run else "")

    if args.once:
        run_once(cfg, gh)
        return

    while True:
        try:
            run_once(cfg, gh)
        except Exception:  # keep the loop alive overnight
            log.exception("Unexpected error during poll")
        log.info("Sleeping %ss… 🦥", cfg.poll_interval)
        time.sleep(cfg.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye 🦥")