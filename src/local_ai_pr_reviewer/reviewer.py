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
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from local_ai_pr_reviewer.promts.promt import SYSTEM_PROMPT, build_user_prompt

GITHUB_API = "https://api.github.com"
MARKER = "<!-- slothreview:{sha} -->"
MAX_REVIEW_BODY = 60_000  # GitHub's limit is 65,536 characters
SEVERITIES = ("critical", "suggestion", "nit")
SEVERITY_HEADINGS = {
    "critical": "🔴 Critical",
    "suggestion": "🟡 Suggestions",
    "nit": "⚪ Nits",
}

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

    def post_comment_review(self, repo: str, number: int, sha: str, body: str) -> None:
        # The event is always COMMENT. This tool never approves or blocks a PR.
        self._request(
            "POST",
            f"/repos/{repo}/pulls/{number}/reviews",
            json={"commit_id": sha, "body": body, "event": "COMMENT"},
        )



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


def clean_issues(raw) -> list:
    issues = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("problem", "")).strip():
            continue
        severity = str(item.get("severity", "suggestion")).lower().strip()
        line = item.get("line")
        issues.append({
            "severity": severity if severity in SEVERITIES else "suggestion",
            "line": line if isinstance(line, int) and line > 0 else None,
            "problem": str(item.get("problem", "")).strip(),
            "suggestion": str(item.get("suggestion", "")).strip(),
            "code": str(item.get("code", "")).strip(),
        })
    return issues


def ask_ollama(cfg: Config, user_prompt: str) -> dict:
    response = requests.post(
        f"{cfg.ollama_url}/api/chat",
        json={
            "model": cfg.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_ctx": cfg.num_ctx},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=900,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    data = parse_model_json(content)
    return {"summary": str(data.get("summary", "")).strip(), "issues": clean_issues(data.get("issues"))}


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
    result = ask_ollama(cfg, prompt)
    result["truncated"] = truncated
    return result


def format_review(cfg: Config, sha: str, results: list, skipped: list) -> str:
    lines = [
        "## 🦥 SlothReview",
        f"> Automated review by a local open-weight model (`{cfg.model}`). "
        "Suggestions only; use your own judgement.",
        "",
    ]

    all_issues = [(f["filename"], issue) for f, result in results for issue in result["issues"]]
    counts = {s: sum(1 for _, i in all_issues if i["severity"] == s) for s in SEVERITIES}

    if not all_issues:
        lines.append("No significant issues found in the reviewed files. Back to napping. 🌿")
    else:
        lines.append(
            f"**{counts['critical']} critical · {counts['suggestion']} suggestions · {counts['nit']} nits**"
        )

    for severity in SEVERITIES:
        group = [(name, i) for name, i in all_issues if i["severity"] == severity]
        if not group:
            continue
        lines += ["", f"### {SEVERITY_HEADINGS[severity]}"]
        for name, issue in group:
            where = f"`{name}`" + (f" (around line {issue['line']})" if issue["line"] else "")
            lines += ["", f"**{where}**: {issue['problem']}"]
            if issue["suggestion"]:
                lines.append(f"\n💡 {issue['suggestion']}")
            if issue["code"]:
                lang = FENCE_LANG.get(Path(name).suffix.lower(), "")
                lines += ["", f"````{lang}", issue["code"], "````"]

    summaries = [(f["filename"], r) for f, r in results if r["summary"]]
    if summaries:
        lines += ["", "<details><summary>Per-file notes</summary>", ""]
        for name, r in summaries:
            flag = " _(diff truncated)_" if r.get("truncated") else ""
            lines.append(f"- `{name}`: {r['summary']}{flag}")
        lines += ["", "</details>"]

    if skipped:
        lines += ["", "<details><summary>Skipped files</summary>", ""]
        lines += [f"- `{name}`: {reason}" for name, reason in skipped]
        lines += ["", "</details>"]

    marker = MARKER.format(sha=sha)
    body = "\n".join(lines)
    if len(body) > MAX_REVIEW_BODY:
        body = body[:MAX_REVIEW_BODY] + "\n\n_…review truncated to fit GitHub's size limit._"
    return f"{body}\n\n{marker}"


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

    body = format_review(cfg, sha, results, skipped)
    if cfg.dry_run:
        print(f"\n===== DRY RUN: {key} =====\n{body}\n")
    else:
        gh.post_comment_review(repo, number, sha, body)
        log.info("Posted review on %s.", key)

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
                log.error("Failed on %s#%s: %s", repo, pr.get("number"), exc)
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