"""Guardrails for SlothReview: everything the model says is checked here before it reaches GitHub.

1. Shape     - pydantic schemas are sent to Ollama as structured output and used to validate replies.
2. Grounding - "current_code" must really be in the diff and "line" must be a changed line.
3. Secrets   - tokens, keys and passwords are masked before anything is posted.
4. Injection - @mentions and HTML comments are neutralised so the bot cannot ping people
               or forge its own de-duplication marker.
5. Noise     - at most MAX_ISSUES_PER_FILE issues per file, most severe first.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger("slothreview")

SEVERITIES = ("critical", "suggestion", "nit")
MAX_ISSUES_PER_FILE = 8
MAX_TEXT = 1_500
MAX_CODE = 2_000


# --------------------------------------------------------------------------- schemas

class Issue(BaseModel):
    severity: Literal["critical", "suggestion", "nit"] = "suggestion"
    title: str = ""
    line: int | None = None
    problem: str
    current_code: str = ""
    suggestion: str = ""
    code: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, value):
        value = str(value or "").lower().strip()
        return value if value in SEVERITIES else "suggestion"

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @field_validator("title", "problem", "current_code", "suggestion", "code", mode="before")
    @classmethod
    def _text(cls, value):
        return "" if value is None else str(value).strip()

    @field_validator("problem")
    @classmethod
    def _problem_required(cls, value):
        if not value:
            raise ValueError("problem is empty")
        return value


class FileReview(BaseModel):
    summary: str = ""
    issues: list[Issue] = Field(default_factory=list)


class Overview(BaseModel):
    overview: str = ""


# Passed to Ollama's "format" field so the model is forced into this shape.
FILE_REVIEW_SCHEMA = FileReview.model_json_schema()
OVERVIEW_SCHEMA = Overview.model_json_schema()


# --------------------------------------------------------------------------- secrets

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.DOTALL),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWT
]
# key = "value" style assignments: keep the key, mask the value.
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (\b[\w.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|auth)[\w.-]*\b
     ["']?\s*[:=]\s*["']?)
    ([^\s"'`,;]{6,})
    """
)
REDACTED = "‹redacted›"


def redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return SECRET_ASSIGNMENT.sub(lambda m: m.group(1) + REDACTED, text)


# --------------------------------------------------------------------------- output safety

def safe_text(text: str, limit: int = MAX_TEXT) -> str:
    """For prose fields: mask secrets, stop @mentions, and drop HTML comments."""
    text = redact_secrets(text)[:limit]
    text = re.sub(r"(?<![\w`])@(?=[A-Za-z0-9])", "@​", text)  # zero-width space: no ping
    return text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")


def safe_code(code: str, limit: int = MAX_CODE) -> str:
    """For code fields: mask secrets and make sure the snippet cannot close our ```` fence."""
    code = redact_secrets(code)[:limit]
    return re.sub(r"`{4,}", "```", code)


# --------------------------------------------------------------------------- grounding

def changed_lines(patch: str) -> set:
    """Line numbers (in the new file) of every added line in a unified diff."""
    lines, current = set(), 0
    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            current = int(hunk.group(1))
        elif raw.startswith("+"):
            lines.add(current)
            current += 1
        elif raw.startswith("-") or raw.startswith("\\"):
            continue
        else:
            current += 1
    return lines


def right_side_lines(patch: str) -> list:
    """(new-file line number, text) for every added or context line, in diff order.
    These are the only lines GitHub accepts an inline comment on (side=RIGHT)."""
    lines, current = [], 0
    for raw in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if hunk:
            current = int(hunk.group(1))
        elif raw.startswith("-") or raw.startswith("\\"):
            continue
        else:
            lines.append((current, raw[1:]))
            current += 1
    return lines


def locate_code(snippet: str, patch: str) -> tuple | None:
    """Find the snippet as consecutive lines on the new side of the diff.
    Returns (start_line, end_line) in the new file, or None if it is not there."""
    wanted = [_normalise(line.lstrip("+")) for line in snippet.splitlines() if line.strip()]
    if not wanted:
        return None
    side = [(number, _normalise(text)) for number, text in right_side_lines(patch) if text.strip()]
    for start in range(len(side) - len(wanted) + 1):
        if all(side[start + k][1] == wanted[k] for k in range(len(wanted))):
            return side[start][0], side[start + len(wanted) - 1][0]
    return None


def _normalise(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def code_in_diff(snippet: str, patch: str) -> bool:
    """True when every non-blank line of the snippet appears in the diff (whitespace-insensitive)."""
    diff_lines = {_normalise(raw[1:]) for raw in patch.splitlines() if raw[:1] in "+- "}
    wanted = [_normalise(line.lstrip("+")) for line in snippet.splitlines() if line.strip()]
    return bool(wanted) and all(line in diff_lines for line in wanted)


# --------------------------------------------------------------------------- entry points

def check_file_review(raw: dict, patch: str, filename: str) -> dict:
    """Validate, ground and sanitise one file's model reply. Bad issues are dropped, not fatal."""
    added = changed_lines(patch)
    issues = []
    for item in raw.get("issues") if isinstance(raw.get("issues"), list) else []:
        try:
            issue = Issue.model_validate(item if isinstance(item, dict) else {})
        except ValidationError:
            log.debug("  guardrail: dropped malformed issue in %s", filename)
            continue

        # Models often copy the diff's "+" markers; show the code as it is in the file.
        issue.current_code = "\n".join(re.sub(r"^\+", "", line) for line in issue.current_code.splitlines())
        if issue.current_code and not code_in_diff(issue.current_code, patch):
            log.debug("  guardrail: current_code not in diff for %s; removed it", filename)
            issue.current_code = ""

        # Anchor the comment: trust where the quoted code really is over the model's line number.
        span = locate_code(issue.current_code, patch) if issue.current_code else None
        if span:
            start_line, line = span
        elif issue.line in added:
            start_line, line = issue.line, issue.line
        else:
            start_line = line = None

        issues.append({
            "severity": issue.severity,
            "title": safe_text(issue.title, 150),
            "start_line": start_line,
            "line": line,
            "problem": safe_text(issue.problem),
            "current_code": safe_code(issue.current_code),
            "suggestion": safe_text(issue.suggestion),
            "code": safe_code(issue.code),
        })

    order = {s: n for n, s in enumerate(SEVERITIES)}
    issues.sort(key=lambda i: order[i["severity"]])
    if len(issues) > MAX_ISSUES_PER_FILE:
        log.info("  guardrail: kept %s of %s issues for %s", MAX_ISSUES_PER_FILE, len(issues), filename)
        issues = issues[:MAX_ISSUES_PER_FILE]

    return {"summary": safe_text(str(raw.get("summary") or "").strip(), 500), "issues": issues}


def check_overview(raw: dict) -> str:
    try:
        overview = Overview.model_validate(raw).overview
    except ValidationError:
        return ""
    return safe_text(str(overview).strip(), 3_000)
