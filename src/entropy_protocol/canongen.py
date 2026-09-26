"""Canonical generation: sealed text is written, token by token, exactly as the verifier will re-read it.

The canonical form (canon.py, v4) folds typographic characters, deletes invisible ones and strips whitespace at the
ends of lines and of the text, so that a text pasted through a word processor still verifies. A verifier or a
detector tokenizes the canonical text; where that differs from the tokens the seal enforced, the checkpoints near
the difference are read in another context. Canonical generation keeps the two identical:

- no character the canonical form changes or deletes (curly quotes, apostrophe letters, typographic dashes, the
  ellipsis character, guillemets, no-break and zero-width spaces, BOM, soft hyphen, word joiner, carriage return);
- valid UTF-8 throughout. Tokens are byte strings and a character may be split across tokens, so the guard tracks
  the unfinished character at the end of the text: inside it only continuation bytes follow, and the text does not
  end there. A token never stops inside a forbidden character past its lead byte; and where the verifier's first
  choice is a banned token, no token may stop inside any character at that step, or a byte-level tokenizer would
  complete the half-written quote into a stray neighbour (a dagger, a star);
- no space or tab before a line break (inside a token, or a line break right after a token that ends in one);
- the first token taken as the verifier reads it once leading whitespace is stripped (``lead``);
- no join the verifier would re-tokenize differently (``joinable``); a HIGH checkpoint never takes a byte prefix
  of the first choice (``high_exclusion``), which the model would complete into a token read back as one.

Where the verifier's first choice is a token canonical generation may not emit, a LOW checkpoint is left free; a
detector recognizes these positions from the text and the model alone (``unsatisfiable``) and skips them, so its
exact null distribution is kept.
"""
from __future__ import annotations

import codecs
import re

import torch

# every character canonical form v4 changes or deletes, written as code points (the source stays ASCII);
# long variation-selector runs are left out (a model does not write them) and single selectors are content
FORBIDDEN_CODEPOINTS = (0x000D, 0xFEFF, 0x200B, 0x2060, 0x00AD, 0x00A0, 0x2007, 0x202F,
                        0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB, 0x2018, 0x2019, 0x201A, 0x201B, 0x02BC,
                        0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0x2026)
# and the replacement character, never content: a model trained on scraped text writes it for a banned apostrophe
# ("don" + U+FFFD + "t" is how mangled web pages spell it)
FORBIDDEN_CODEPOINTS += (0xFFFD,)
FORBIDDEN = tuple(chr(c).encode("utf-8") for c in FORBIDDEN_CODEPOINTS)
PREFIXES = sorted({f[:j] for f in FORBIDDEN for j in range(1, len(f))})     # tracked exactly
PREFIX_SET = frozenset(PREFIXES)
DANGEROUS = frozenset(f[:j] for f in FORBIDDEN for j in range(2, len(f)))  # a token never stops here
# stand-ins for "inside some other character, n bytes to go" (none is a prefix of a forbidden character)
INSIDE = {1: bytes([0xE3, 0x80]), 2: bytes([0xE3]), 3: bytes([0xF0])}
LINE_END_WS = (b" \n", b"\t\n", b" \r", b"\t\r")
# continuations tried when a join is checked: end of text, a letter, a space and a letter, a line break, a full
# stop, a digit. Some tokenizers split a whitespace run by what follows it (GPT-2 reads "\n\nThe" as three tokens
# but "\n\n The" as two), so a join is kept when some continuation leaves it intact.
PROBES = (b"", b"x", b" x", b"\n", b".", b"1")


def _bytes_to_unicode() -> dict[int, str]:
    """The byte <-> printable-character table of byte-level BPE (GPT-2, Llama 3, Qwen)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def token_bytes(tok) -> list[bytes | None]:
    """Raw bytes of every token id; None for special tokens (they never enter the text)."""
    special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if getattr(t, "special", False)}
    vocab = tok.get_vocab()
    n = max(vocab.values()) + 1
    pieces = [None] * n
    for piece, i in vocab.items():
        pieces[i] = piece
    space_marker, sp_marker = chr(0x120), chr(0x2581)                     # GPT-2 and SentencePiece space marks
    byte_level = any(space_marker in p for p in pieces[:5000] if p)
    dec = {c: b for b, c in _bytes_to_unicode().items()}
    fallback = re.compile(r"<0x([0-9A-Fa-f]{2})>")
    out: list[bytes | None] = []
    for i, p in enumerate(pieces):
        if p is None or i in special:
            out.append(None)
        elif byte_level:
            out.append(bytes(dec[c] for c in p) if all(c in dec for c in p) else p.encode("utf-8"))
        else:
            m = fallback.fullmatch(p)
            out.append(bytes([int(m.group(1), 16)]) if m else p.replace(sp_marker, " ").encode("utf-8"))
    return out


def _tail(b: bytes) -> bytes:
    """The incomplete UTF-8 character at the end of ``b`` (empty if it ends on a character boundary)."""
    for k in range(1, min(4, len(b)) + 1):
        c = b[-k]
        if c & 0xC0 != 0x80:                       # a lead byte (or ASCII): the last character starts here
            need = 1 if c < 0x80 else 2 if c >> 5 == 0x6 else 3 if c >> 4 == 0xE else 4 if c >> 3 == 0x1E else 1
            return b[-k:] if k < need else b""
    return b""


def _need(tail: bytes) -> int:
    """Continuation bytes still missing from an unfinished character."""
    c = tail[0]
    total = 2 if c >> 5 == 0x6 else 3 if c >> 4 == 0xE else 4 if c >> 3 == 0x1E else 1
    return max(0, total - len(tail))


def _utf8_ok(b: bytes) -> bool:
    """No invalid sequence in ``b`` (an unfinished character at its end is allowed)."""
    try:
        codecs.getincrementaldecoder("utf-8")().decode(b, final=False)
        return True
    except UnicodeDecodeError:
        return False


class CanonGuard:
    """Per-step ban masks and checks that keep generated tokens identical to the verifier's reading."""

    WINDOW = 8          # tokens of context re-tokenized when a join is checked
    COMPARE = 4         # of which the last COMPARE must come back unchanged (the window's left edge may not)

    def __init__(self, tok, vocab_size: int, device="cpu"):
        self.tok = tok
        self.bytes = token_bytes(tok)
        # states: at a character boundary, inside one of the forbidden characters' prefixes (tracked exactly), or
        # inside some other character with n bytes to go
        self.states: list[bytes] = [b""] + PREFIXES + [INSIDE[n] for n in (1, 2, 3)]
        self.index = {s: i for i, s in enumerate(self.states)}
        n_tok = min(vocab_size, len(self.bytes))
        masks = torch.zeros((len(self.states), vocab_size), dtype=torch.bool)
        masks[:, n_tok:] = True                    # embedding rows without a token: never emitted
        partial = torch.zeros(vocab_size, dtype=torch.bool)
        nl = torch.zeros(vocab_size, dtype=torch.bool)
        for i in range(n_tok):
            tb = self.bytes[i]
            if tb is None:
                masks[1:, i] = True                # a special token (the end of the text) never inside a character
                continue
            if not tb or any(w in tb for w in LINE_END_WS):
                masks[:, i] = True                 # no text, or a space or tab before a line break
                continue
            nl[i] = tb[:1] in (b"\n", b"\r")
            partial[i] = bool(_tail(tb))
            plain = max(tb) < 0x80 and b"\r" not in tb
            for si, s in enumerate(self.states):
                if not s and plain:
                    continue                       # plain ASCII at a boundary: always fine
                joined = s + tb
                if not _utf8_ok(joined) or any(f in joined for f in FORBIDDEN) or _tail(joined) in DANGEROUS:
                    masks[si, i] = True
        self.masks, self.masks_cpu = masks.to(device), masks
        self.partial, self.partial_cpu = partial.to(device), partial         # tokens that stop inside a character
        self.nl_start, self.nl_start_cpu = nl.to(device), nl                 # banned after a trailing space or tab
        self.banned_any = int(masks[0, :n_tok].sum())
        self.lead = self._lead_map(tok, vocab_size)          # CPU: verifier-view first token, -1 = not first
        self.lead_ban = (self.lead < 0).to(device)
        self.prefixes_of = self._prefix_table(n_tok)
        self._cache: dict[tuple, bool] = {}

    # ---------------------------------------------------------------------------------------------- tables
    def _lead_map(self, tok, vocab_size: int) -> torch.Tensor:
        """The token the verifier reads for a first token once leading whitespace is stripped; -1 where there is no
        single such token (pure whitespace, a remainder of several tokens or not whole UTF-8)."""
        lead = torch.arange(vocab_size, dtype=torch.long)
        lead[len(self.bytes):] = -1                        # embedding rows without a token
        todo_ids: list[int] = []
        todo_text: list[str] = []
        for i, tb in enumerate(self.bytes[:vocab_size]):
            if tb is None or not tb:
                continue                                   # special tokens do not enter the text
            try:
                s = tb.decode("utf-8")
            except UnicodeDecodeError:                     # a piece of a character
                if tb[:1].isspace() or tb[0] & 0xC0 == 0x80:
                    lead[i] = -1                           # unreadable once stripped / no character starts here
                continue
            rem = s.lstrip()
            if rem == s:
                continue
            if rem:
                todo_ids.append(i)
                todo_text.append(rem)
            else:
                lead[i] = -1                               # pure whitespace: stripped away entirely
        if todo_text:
            for i, e in zip(todo_ids, tok(todo_text, add_special_tokens=False)["input_ids"]):
                lead[i] = e[0] if len(e) == 1 else -1
        return lead

    def _prefix_table(self, n_tok: int) -> dict[int, list[int]]:
        """Token -> the tokens whose bytes are a proper prefix of its bytes. A HIGH checkpoint must not pick a prefix
        of the first choice (" question" for " questionnaire", "." for a full stop with line breaks): the model
        would complete the word and the verifier re-read the merged token."""
        by_bytes: dict[bytes, list[int]] = {}
        for i, tb in enumerate(self.bytes[:n_tok]):
            if tb:
                by_bytes.setdefault(tb, []).append(i)
        table: dict[int, list[int]] = {}
        for i, tb in enumerate(self.bytes[:n_tok]):
            if tb and len(tb) > 1:
                pre = [t for m in range(1, len(tb)) for t in by_bytes.get(tb[:m], ())]
                if pre:
                    table[i] = pre
        # and the tokens that only add whitespace to it ("\n\n" for "\n", ".\n\n" for "."): where a whitespace run
        # ends is decided by what follows it, so a HIGH checkpoint must not move that boundary either
        for i, tb in enumerate(self.bytes[:n_tok]):
            if not tb or not tb[-1:].isspace():
                continue
            core = len(tb.rstrip())                        # tb = tb[:core] + a whitespace run
            for m in range(max(1, core), len(tb)):
                for t in by_bytes.get(tb[:m], ()):
                    table.setdefault(t, []).append(i)
        return table

    # ------------------------------------------------------------------------------------------------- steps
    def mask(self, state: int) -> torch.Tensor:
        return self.masks[state]

    def state_after(self, state: int, token_id: int) -> int:
        tb = self.bytes[token_id] if token_id < len(self.bytes) else None
        if not tb:
            return 0
        tail = _tail(self.states[state] + tb)
        if not tail:
            return 0
        if tail in PREFIX_SET:
            return self.index[tail]
        n = _need(tail)
        return self.index[INSIDE[n]] if n in INSIDE else 0

    def ends_in_space(self, token_id: int) -> bool:
        """Does the token end in a space or tab (so that a line break may not follow)?"""
        tb = self.bytes[token_id] if token_id < len(self.bytes) else None
        return bool(tb) and tb[-1:] in (b" ", b"\t")

    def high_exclusion(self, first: list[int]) -> list[int]:
        """Tokens a HIGH checkpoint must not choose, given the first choice(s) it has to avoid."""
        return [t for f in first for t in self.prefixes_of.get(int(f), ())]

    def joinable(self, view: list[int], t: int) -> bool:
        """Would the verifier re-read the text of ``view`` + [t] as these very tokens, for some continuation of the
        text? The last tokens are re-tokenized with each probe continuation; the last COMPARE tokens of the window
        and t must come back unchanged, followed by exactly the probe. A join inside an unfinished character is
        judged when the character is complete. A break already in the window (a join kept when nothing else
        joined) is not held against the tokens after it."""
        tb = self.bytes[t] if t < len(self.bytes) else None
        if tb is None or not view:
            return True
        start = len(view) <= self.WINDOW                  # the window begins where the text begins
        if len(view) > 1 and not self._intact(tuple(view[-self.WINDOW:]), start):
            return True
        return self._intact(tuple(view[-self.WINDOW:]) + (t,), start)

    def _intact(self, win: tuple, start: bool) -> bool:
        hit = self._cache.get((win, start))
        if hit is None:
            hit = self._check(list(win), start)
            if len(self._cache) > 50000:
                self._cache.clear()
            self._cache[(win, start)] = hit
        return hit

    def _check(self, seq: list[int], start: bool = False) -> bool:
        """Do the last COMPARE tokens of ``seq`` come back unchanged when its text is re-tokenized with some probe?
        At the start of a text a leading space is the tokenizer's own prefix (SentencePiece), never text: the
        canonical text does not begin with whitespace."""
        win, t = seq[:-1], seq[-1]
        tb = self.bytes[t] if t < len(self.bytes) else None
        if tb is None:
            return True
        parts = [self.bytes[i] if i < len(self.bytes) else None for i in win]
        if any(p is None for p in parts):
            return True
        body = b"".join(parts) + tb
        if start and body[:1] == b" ":
            body = body[1:]
        cut = 0
        while cut < len(body) and body[cut] & 0xC0 == 0x80:      # a character cut by the window's left edge
            cut += 1
        body = body[cut:]
        if _tail(body):
            return True
        want = win[-self.COMPARE:] + [t]
        try:
            texts = [(body + p).decode("utf-8") for p in PROBES]
        except UnicodeDecodeError:
            return True
        for p, enc in zip(PROBES, self.tok(texts, add_special_tokens=False)["input_ids"]):
            for q in range(3):
                end = len(enc) - q
                if end < len(want) or enc[end - len(want):end] != want:
                    continue
                if b"".join(self.bytes[i] or b"" for i in enc[end:]) == p:
                    return True
        return False

    def first_joinable(self, view: list[int], scores: torch.Tensor, avoid, stops, limit: int = 64) -> int | None:
        """The most probable token under ``scores`` (among the ``limit`` best, none in ``avoid`` or ``stops``) that
        joins ``view`` as the verifier will read it; None if there is none."""
        for t in torch.argsort(scores, descending=True)[:limit].tolist():
            if scores[t] == float("-inf"):
                break
            if t not in avoid and t not in stops and self.joinable(view, t):
                return t
        return None

    def unsatisfiable(self, ids: list[int], positions: list[int], first: list[int], stops) -> list[bool]:
        """At each position (a token index of ``ids``): could canonical generation not have emitted the verifier's
        first choice ``first`` there? (A stop token, a token the guard bans after the text so far, or a join the
        verifier would re-read differently.) The sealer then left a LOW checkpoint free. Read from the text and the
        model, never from the key, so a detector that skips these positions keeps the exact null."""
        states, spaces, s, sp = [], [], 0, False
        for t in ids[: max(positions, default=-1) + 1]:
            states.append(s)
            spaces.append(sp)
            s, sp = self.state_after(s, t), self.ends_in_space(t)
        out = []
        for p, f in zip(positions, first):
            if f in stops or f >= self.masks_cpu.shape[1]:
                out.append(True)
            elif bool(self.masks_cpu[states[p], f]) or (spaces[p] and bool(self.nl_start_cpu[f])):
                out.append(True)
            elif p == 0:
                out.append(bool(self.lead[f] < 0))
            else:
                out.append(f != ids[p] and not self.joinable(ids[:p], f))
        return out
