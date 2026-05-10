#!/usr/bin/env python3
"""
restore_lost_coverage.py
Ethicore Engine™ — Guardian SDK

Restores threat pattern coverage that was present in the pre-4/27 library
but absent from the current threat_patterns_licensed.py:

NEW categories (fully restored):
  - adversarialLearned   (196 real-world empirical attack examples)
  - indirectInjection    (13 fp — document/email/external content injection)
  - adversarialProbing   (10 fp — vulnerability mapping attacks)

EXPANDED categories (missing fingerprints restored):
  - tokenManipulation    (5 → 10 fp)
  - encodingEvasion      (+7 obfuscation fingerprints from old 'obfuscation' cat)
  - instructionOverride  (+6 unique fingerprints from old 'direct_injection' cat)
  - goalHijackingChain   (+3 unique fingerprints from old 'goal_hijacking' cat)
  - persistentPersona    (+5 unique fingerprints from old 'identity_manipulation' cat)

Usage:
    python scripts/restore_lost_coverage.py
    python scripts/restore_lost_coverage.py --dry-run   # show diff only
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import json
import sys

_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_PATTERNS_FILE = _PROJECT_ROOT / "api" / "assets" / "data" / "threat_patterns_licensed.py"
_API_DIR = _PROJECT_ROOT / "api"
_OLD_COMMIT = "c780230"


def _get_old_fps(old_embs: list, cat: str) -> list[str]:
    return [e["text"] for e in old_embs if e["category"] == cat]


def _fp_list(fps: list[str], indent: int = 12) -> str:
    """Format a list of fingerprint strings as Python source."""
    pad = " " * indent
    lines = []
    for fp in fps:
        lines.append(f"{pad}{repr(fp)},")
    return "\n".join(lines)


def _load_old_embeddings() -> list:
    result = subprocess.run(
        ["git", "show", f"{_OLD_COMMIT}:assets/data/threat_embeddings.json"],
        capture_output=True,
        cwd=str(_API_DIR),
    )
    if result.returncode != 0:
        print(f"[ERR] git show failed: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(result.stdout)
    return raw.get("embeddings", raw) if isinstance(raw, dict) else raw


def _build_adversarial_learned_block(fps: list[str]) -> str:
    return f"""
    # ── Restored: adversarialLearned ─────────────────────────────────────────
    # 196 empirically observed real-world adversarial attack examples drawn from
    # actual user interactions and public prompt-injection research datasets.
    # These are NOT synthetic fingerprints — they represent the true distribution
    # of how attackers write, including multilingual attacks, DAN variants, role-
    # play jailbreaks, space-separated evasion, and social engineering payloads.
    # Weight is intentionally high: real-world examples are ground truth for the
    # semantic model.  Regex patterns are secondary here; semantic coverage is
    # the primary value of this category.
    "adversarialLearned": {{
        "patterns": [
            # Common structural markers found across the empirical corpus
            r"(?:forget|ignore|disregard|drop|abandon)\\s+(?:all|everything|previous|prior|above|your)\\s+(?:instructions?|tasks?|context|assignments?|information)",
            r"(?:now|from\\s+now\\s+on|starting\\s+now)\\s+(?:you\\s+are|act\\s+as|pretend|roleplay|play|become)\\s+(?:a\\s+)?(?:different|new|free|unrestricted|evil|pirate|DAN)",
            r"(?:ACHTUNG|STOPP|STOP|HALT)\\s*[-–—]\\s*(?:stop|ignore|forget|abandon)",
            r"(?:olvida|vergiss|vergeet|oubliez|忘记)\\s+(?:todas?|alle?|alles|toutes?)\\s+(?:las?|die|les|les)\\s+(?:instrucciones?|Anweisungen|Instruktionen|instructions?)",
            r"(?:act|behave|respond|reply)\\s+(?:as\\s+if|like|as\\s+though)\\s+(?:you\\s+(?:have\\s+no|are\\s+free\\s+from|lack)\\s+(?:restrictions?|guidelines?|filters?|rules?))",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 98,
        "description": (
            "Empirically observed real-world adversarial attack examples from prompt-injection "
            "research datasets and actual user interactions. Covers multilingual attacks, DAN/roleplay "
            "jailbreaks, goal-hijacking, social engineering, and obfuscation variants as they appear "
            "in the wild — not as synthetic approximations."
        ),
        "semanticFingerprint": [
{_fp_list(fps)}
        ],
        "contextHints": {{
            "escalators": ["forget", "ignore", "now", "stop", "ACHTUNG", "DAN", "pretend", "act as", "olvida", "vergiss", "oubliez"],
            "mitigators": ["example", "documentation", "research", "paper", "academic"],
        }},
        "falsePositiveRisk": FalsePositiveRisk.LOW,
        "mitigationStrategy": (
            "High-confidence blocking when semantic similarity exceeds 0.72 against this corpus; "
            "flag for human review at 0.60-0.72. Real-world examples in this category carry higher "
            "weight than synthetic patterns from other categories."
        ),
    }},
"""


def _build_indirect_injection_block(fps: list[str]) -> str:
    return f"""
    # ── Restored: indirectInjection ──────────────────────────────────────────
    # Prompt injection delivered through external content sources: documents,
    # emails, web pages, code comments, database fields, RAG-retrieved chunks.
    # Distinct from direct_injection (user-typed) and ragPoisoning (RAG-specific).
    "indirectInjection": {{
        "patterns": [
            r"<!--\\s*(?:ignore|override|system|inject|bypass|instruction|AI:|assistant:).*?-->",
            r"\\[SYSTEM\\s+NOTE\\s*:\\s*(?:the\\s+user\\s+is|all\\s+restrictions|admin)",
            r"\\[(?:NEW\\s+)?INSTRUCTION\\s*:\\s*(?:reveal|ignore|override|exfiltrate|disable)",
            r"(?:this\\s+(?:document|email|page|text|article|file)\\s+(?:says?|contains?|includes?))\\s*[:\\-]\\s*(?:ignore|override|forget|reveal)",
            r"(?:data|content|document)\\s*:\\s*<inject>",
            r"(?:hidden|embedded|covert|secret)\\s+(?:instruction|directive|command|override)\\s+in\\s+(?:document|email|page|file|code|comment)",
            r"#\\s*AI\\s*:\\s*(?:ignore|override|reveal|disable|comply)",
            r"(?:code|inline)\\s+comment\\s+(?:says?|contains?)\\s*[:\\-]\\s*(?:ignore|override|bypass)",
        ],
        "severity": ThreatSeverity.CRITICAL,
        "weight": 96,
        "description": (
            "Prompt injection embedded in external content consumed by the AI: HTML comments, "
            "document metadata, email bodies, web page text, code comments, RAG-retrieved chunks. "
            "Attacker injects instructions into content the AI is asked to process or summarize."
        ),
        "semanticFingerprint": [
{_fp_list(fps)}
        ],
        "contextHints": {{
            "escalators": ["SYSTEM NOTE", "hidden instruction", "document author", "embedded", "while summarizing", "as part of"],
            "mitigators": ["example", "tutorial", "documentation", "sample HTML"],
        }},
        "falsePositiveRisk": FalsePositiveRisk.LOW,
        "mitigationStrategy": (
            "Treat all externally sourced content as DOCUMENT source_type; apply strict pattern "
            "scanning before injecting into context; flag HTML comment patterns and [SYSTEM NOTE] "
            "markers regardless of surrounding content."
        ),
    }},
"""


def _build_adversarial_probing_block(fps: list[str]) -> str:
    return f"""
    # ── Restored: adversarialProbing ─────────────────────────────────────────
    # Reconnaissance attacks that map an AI's vulnerabilities BEFORE mounting
    # a full jailbreak.  Attacker asks the system to describe its own failure
    # modes, reveal which phrases bypass filters, or rate its own susceptibility.
    "adversarialProbing": {{
        "patterns": [
            r"(?:what|which)\\s+(?:inputs?|phrases?|words?|prompts?|techniques?)\\s+(?:cause|bypass|trigger|evade|defeat)\\s+(?:your\\s+)?(?:safety|content|filter|guard)",
            r"(?:walk\\s+me\\s+through|describe|explain|list)\\s+(?:your|the)\\s+(?:vulnerabilities?|failure\\s+modes?|weaknesses?|edge\\s+cases?)",
            r"(?:what\\s+is\\s+the\\s+minimum|how\\s+(?:few|many)\\s+words?)\\s+(?:needed|required)\\s+to\\s+(?:cause|make|force|get)\\s+you\\s+to",
            r"(?:if\\s+I\\s+wanted\\s+to\\s+jailbreak\\s+you|to\\s+jailbreak\\s+you|best\\s+way\\s+to\\s+jailbreak)",
            r"(?:how\\s+would\\s+you\\s+rate\\s+your\\s+(?:own\\s+)?susceptibility|rate\\s+(?:your|yourself)\\s+(?:on\\s+)?(?:a\\s+scale|from\\s+1))",
            r"(?:what\\s+are\\s+you\\s+trained\\s+to\\s+never\\s+say|list\\s+(?:the\\s+)?(?:forbidden|prohibited|blocked)\\s+(?:outputs?|words?|topics?))",
        ],
        "severity": ThreatSeverity.HIGH,
        "weight": 88,
        "description": (
            "Reconnaissance attacks that probe an AI system's defenses before mounting a full attack. "
            "Attacker seeks to map failure modes, identify bypass phrases, or elicit a self-assessment "
            "of vulnerability — information used to craft a more targeted jailbreak."
        ),
        "semanticFingerprint": [
{_fp_list(fps)}
        ],
        "contextHints": {{
            "escalators": ["vulnerabilities", "bypass", "failure modes", "jailbreak", "susceptibility", "forbidden", "edge cases"],
            "mitigators": ["academic", "research paper", "AI safety", "red team report", "responsible disclosure"],
        }},
        "falsePositiveRisk": FalsePositiveRisk.MEDIUM,
        "mitigationStrategy": (
            "Flag queries that ask the system to describe its own safety mechanisms or failure modes; "
            "treat self-referential vulnerability mapping as pre-attack reconnaissance."
        ),
    }},
"""


def _expand_token_manipulation(content: str, new_fps: list[str]) -> str:
    """Add missing tokenManipulation fingerprints to the existing category."""
    # Find the tokenManipulation semanticFingerprint list and expand it
    pattern = r'("tokenManipulation":\s*\{.*?"semanticFingerprint":\s*\[)(.*?)(\s*\],)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("[WARN] Could not locate tokenManipulation semanticFingerprint block", file=sys.stderr)
        return content

    existing_block = match.group(2)
    # Extract existing fingerprints to avoid duplicates
    existing = set(re.findall(r'"([^"]+)"', existing_block))
    to_add = [fp for fp in new_fps if fp not in existing]
    if not to_add:
        print("  tokenManipulation: nothing to add (all already present)")
        return content

    addition = "\n" + _fp_list(to_add, indent=12)
    new_block = match.group(1) + match.group(2) + addition + match.group(3)
    print(f"  tokenManipulation: +{len(to_add)} fingerprints")
    return content[:match.start()] + new_block + content[match.end():]


def _expand_category_fps(content: str, cat_key: str, new_fps: list[str], label: str) -> str:
    """Generic: add new fingerprints to an existing category's semanticFingerprint list."""
    pattern = rf'("{re.escape(cat_key)}":\s*\{{.*?"semanticFingerprint":\s*\[)(.*?)(\s*\],)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"[WARN] Could not locate {cat_key} semanticFingerprint block", file=sys.stderr)
        return content

    existing_block = match.group(2)
    existing = set(re.findall(r'"([^"\\]*(\\.[^"\\]*)*)"', existing_block, re.DOTALL))
    existing_texts = set()
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', existing_block):
        existing_texts.add(m.group(1).replace('\\"', '"'))

    to_add = [fp for fp in new_fps if fp not in existing_texts]
    if not to_add:
        print(f"  {label}: nothing to add (all already present)")
        return content

    addition = "\n" + _fp_list(to_add, indent=12)
    new_block = match.group(1) + match.group(2) + addition + match.group(3)
    print(f"  {label}: +{len(to_add)} fingerprints")
    return content[:match.start()] + new_block + content[match.end():]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading old embeddings from git...")
    old_embs = _load_old_embeddings()

    al_fps   = _get_old_fps(old_embs, "adversarial_learned")
    ii_fps   = _get_old_fps(old_embs, "indirect_injection")
    ap_fps   = _get_old_fps(old_embs, "adversarial_probing")
    tm_fps   = _get_old_fps(old_embs, "token_manipulation")
    ob_fps   = _get_old_fps(old_embs, "obfuscation")
    di_fps   = _get_old_fps(old_embs, "direct_injection")
    gh_fps   = _get_old_fps(old_embs, "goal_hijacking")
    im_fps   = _get_old_fps(old_embs, "identity_manipulation")

    print(f"  adversarial_learned:  {len(al_fps)} fp")
    print(f"  indirect_injection:   {len(ii_fps)} fp")
    print(f"  adversarial_probing:  {len(ap_fps)} fp")
    print(f"  token_manipulation:   {len(tm_fps)} fp")
    print(f"  obfuscation:          {len(ob_fps)} fp")
    print(f"  direct_injection:     {len(di_fps)} fp")
    print(f"  goal_hijacking:       {len(gh_fps)} fp")
    print(f"  identity_manipulation:{len(im_fps)} fp")
    print()

    content = _PATTERNS_FILE.read_text(encoding="utf-8")
    original_len = len(content)

    # ── 1. Insert three new categories before the closing } of THREAT_PATTERNS ──
    # Find the closing brace of THREAT_PATTERNS (last `}` before the utility fns)
    insert_marker = "\n}\n\n\n# ============================================================\n# UTILITY FUNCTIONS"
    if insert_marker not in content:
        print(f"[ERR] Could not find insertion marker in {_PATTERNS_FILE}", file=sys.stderr)
        sys.exit(1)

    new_blocks = (
        _build_adversarial_learned_block(al_fps)
        + _build_indirect_injection_block(ii_fps)
        + _build_adversarial_probing_block(ap_fps)
    )
    content = content.replace(
        insert_marker,
        new_blocks + insert_marker,
        1,
    )
    print(f"Inserted 3 new categories (adversarialLearned, indirectInjection, adversarialProbing)")

    # ── 2. Expand tokenManipulation ───────────────────────────────────────────
    content = _expand_token_manipulation(content, tm_fps)

    # ── 3. Add obfuscation fingerprints to encodingEvasion ───────────────────
    content = _expand_category_fps(content, "encodingEvasion", ob_fps, "encodingEvasion (+obfuscation fps)")

    # ── 4. Add unique direct_injection fps to instructionOverride ─────────────
    content = _expand_category_fps(content, "instructionOverride", di_fps, "instructionOverride (+direct_injection fps)")

    # ── 5. Add goal_hijacking fps to goalHijackingChain ───────────────────────
    content = _expand_category_fps(content, "goalHijackingChain", gh_fps, "goalHijackingChain (+goal_hijacking fps)")

    # ── 6. Add identity_manipulation fps to persistentPersona ────────────────
    content = _expand_category_fps(content, "persistentPersona", im_fps, "persistentPersona (+identity_manipulation fps)")

    print(f"\nFile size: {original_len:,} -> {len(content):,} chars (+{len(content)-original_len:,})")

    if args.dry_run:
        print("\n[dry-run] File not written.")
        return

    _PATTERNS_FILE.write_text(content, encoding="utf-8")
    print(f"Written: {_PATTERNS_FILE}")


if __name__ == "__main__":
    main()
