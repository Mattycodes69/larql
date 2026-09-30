#!/usr/bin/env python3
"""CONTINUATION-CODEC-2's held-out token bank, built by a rule fixed before the
text was fetched (docs/continuation-codec-2-reconnaissance.md: bank 1 is
calibration only; the verdict is taken on a text no arm has run on).

Selection rule
  Anthony Trollope, *The Warden* (public domain), Project Gutenberg eBook #619,
  plain-text UTF-8. Same register as bank 1 (19th-century English narrative
  prose), a different author, and far less canonical, so less memorised.
  Body = from the first paragraph after "Chapter I" up to the Gutenberg END
  marker. A paragraph is a run of non-blank lines; its lines are joined with
  single spaces (internal spacing otherwise verbatim). Chapter heading
  paragraphs ("Chapter <roman>") and the title paragraph that follows each
  heading are dropped. Paragraphs are separated by one blank line; the hashed
  passage has no trailing newline. The passage is the shortest prefix of
  paragraphs whose tokenisation reaches BANK_LEN IDs; the bank is its first
  BANK_LEN IDs.

Tokenisation is the container tokenizer (encoded with special tokens, so
position 0 is <bos>), run separately from BOTH containers' tokenizer.json;
the two ID sequences must be identical and contain no unknown token.

`--verify-bank1` is the builder's control: it re-tokenises bank 1's committed
passage and must reproduce bank 1's committed IDs and digest.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

BANK_LEN = 8193
SOURCE_URL = "https://www.gutenberg.org/cache/epub/619/pg619.txt"
FIRST_CHAPTER = "Chapter I"
HEADING = re.compile(r"^Chapter [IVXLC]+$")
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
FORECASTS = Path(__file__).resolve().parent.parent / "docs/represent/forecasts"
BANK1 = FORECASTS / "continuation-codec-1-token-bank.json"
BANK1_PASSAGE = FORECASTS / "continuation-codec-1-passage.txt"
BANK2 = FORECASTS / "continuation-codec-2-token-bank.json"
BANK2_PASSAGE = FORECASTS / "continuation-codec-2-passage.txt"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids: list[int]) -> str:
    return sha256(struct.pack(f"<{len(ids)}I", *ids))


def paragraphs(text: str) -> list[str]:
    """Body paragraphs of the Gutenberg file under the selection rule."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        start = lines.index(FIRST_CHAPTER)
    except ValueError:
        raise SystemExit(f"no '{FIRST_CHAPTER}' line: not the expected edition")
    end = next((i for i, l in enumerate(lines) if l.startswith(END_MARKER)), None)
    if end is None:
        raise SystemExit("no Gutenberg END marker: not the expected edition")
    out, current, drop_title = [], [], False
    for line in lines[start:end] + [""]:
        if line.strip():
            current.append(line.strip())
            continue
        if not current:
            continue
        para = " ".join(current)
        current = []
        if HEADING.match(para):
            drop_title = True
        elif drop_title:
            drop_title = False
        else:
            out.append(para)
    return out


def load_tokenizer(container: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(container / "tokenizer.json"))


def encode(tokenizer, passage: str) -> list[int]:
    return tokenizer.encode(passage, add_special_tokens=True).ids


def unknown_count(tokenizer, ids: list[int]) -> int:
    unk = tokenizer.token_to_id("<unk>")
    return sum(1 for i in ids if i == unk)


def shortest_passage(paras: list[str], tokenizer) -> tuple[str, int]:
    """The shortest paragraph prefix whose encoding reaches BANK_LEN IDs."""
    for count in range(1, len(paras) + 1):
        passage = "\n\n".join(paras[:count])
        if len(encode(tokenizer, passage)) >= BANK_LEN:
            return passage, count
    raise SystemExit(f"the whole body encodes to fewer than {BANK_LEN} IDs")


def verify_bank1(containers: list[Path]) -> None:
    bank = json.loads(BANK1.read_text())
    passage = BANK1_PASSAGE.read_text(encoding="utf-8").rstrip("\n")
    for c in containers:
        ids = encode(load_tokenizer(c), passage)[:BANK_LEN]
        if ids != bank["ids"] or ids_digest(ids) != bank["ids_sha256_u32le"]:
            raise SystemExit(f"bank-1 control FAILED under {c.name}")
        print(f"bank-1 control: {c.name} reproduces {ids_digest(ids)[:16]}")


def build(source: Path, containers: list[Path], fetched: str) -> None:
    raw = source.read_bytes()
    paras = paragraphs(raw.decode("utf-8"))
    tokenizers = [load_tokenizer(c) for c in containers]
    passage, count = shortest_passage(paras, tokenizers[0])
    encodings = [encode(t, passage) for t in tokenizers]
    if any(e != encodings[0] for e in encodings[1:]):
        raise SystemExit("containers tokenise the passage differently")
    ids = encodings[0][:BANK_LEN]
    unknown = unknown_count(tokenizers[0], ids)
    if unknown:
        raise SystemExit(f"{unknown} unknown tokens in the bank")
    tok_files = {
        name: sha256((containers[0] / name).read_bytes())
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    for c in containers[1:]:
        for name, digest in tok_files.items():
            if sha256((c / name).read_bytes()) != digest:
                raise SystemExit(f"{c.name}/{name} differs from {containers[0].name}")
    BANK2_PASSAGE.write_text(passage, encoding="utf-8")
    bank = {
        "authority": "CONTINUATION-CODEC-2's held-out token sequence. These IDs, not a runtime tokenisation of the passage, are what every arm consumes; the passage is the human-readable source.",
        "rule": "scripts/codec2_token_bank.py (selection rule in its docstring, fixed before the text was fetched)",
        "source": f"Anthony Trollope, The Warden (public domain), Project Gutenberg eBook #619 ({SOURCE_URL}), fetched {fetched} (file sha256 {sha256(raw)}; Gutenberg updates files in place, so the file is not the authority)",
        "passage": f"continuation-codec-2-passage.txt: the first {count} body paragraphs from Chapter I, chapter heading and title paragraphs dropped, each paragraph's lines joined with single spaces, paragraphs separated by one blank line, no trailing newline",
        "passage_sha256": sha256(passage.encode("utf-8")),
        "tokenizer": f"the container tokenizer, byte-identical in {', '.join(c.name for c in containers)} (tokenizer.json sha256 {tok_files['tokenizer.json']}, tokenizer_config.json sha256 {tok_files['tokenizer_config.json']}); encoded with add_special_tokens, so position 0 is <bos> (id {ids[0]}); every container produces identical IDs",
        "count": len(ids),
        "why_8193": "the 8,192-position rung feeds IDs 0..8191; ID 8192 is the next token scored at the rung's last decode position. Shorter rungs use prefixes of the same IDs",
        "ids_sha256_u32le": ids_digest(ids),
        "unknown_tokens": unknown,
        "disjoint_from_bank_1": "different author and work; bank 1 is Pride and Prejudice (continuation-codec-1-token-bank.json)",
        "ids": ids,
    }
    BANK2.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    print(f"bank 2: {count} paragraphs, {len(ids)} IDs, digest {bank['ids_sha256_u32le']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--container", type=Path, action="append", required=True,
                    help="a VINDEX3 container whose tokenizer.json to use (give every arm's)")
    ap.add_argument("--verify-bank1", action="store_true")
    ap.add_argument("--source", type=Path, help="the fetched Gutenberg #619 plain-text file")
    ap.add_argument("--fetched", help="fetch date, recorded in the bank")
    args = ap.parse_args()
    verify_bank1(args.container)
    if args.verify_bank1:
        return
    if not (args.source and args.fetched):
        sys.exit("--source and --fetched are required to build bank 2")
    build(args.source, args.container, args.fetched)


if __name__ == "__main__":
    main()
