import argparse
import math
import mmap
import os
from urllib.request import urlopen
from typing import Dict, List, Tuple

CACHE_FILE = "pi_digits.txt"
DEFAULT_SOURCE = "https://stuff.mit.edu/afs/sipb/contrib/pi/pi-billion.txt"
DEFAULT_WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
DEFAULT_WORDLIST_CACHE = "words_alpha.txt"

NON_DIGITS = bytes([b for b in range(256) if not (48 <= b <= 57)])
IDENTITY = bytes.maketrans(b"", b"")

# Optional speedup table for encoding text -> "DDD" digits
TRIPLES = [f"{i:03d}".encode("ascii") for i in range(256)]

# Letter-to-2-digit mapping for letters2 encoding (a-z -> 01..26)
_LETTERS2 = {chr(ord('a') + i): f"{i+1:02d}" for i in range(26)}

def text_to_utf8_digits(text: str) -> bytes:
    b = text.encode('utf-8')
    return b"".join(TRIPLES[x] for x in b)

def text_to_letters2_digits(text: str) -> bytes:
    """Map letters a-z/A-Z -> 01..26 (two digits per letter). Raises on non-letters."""
    s = text.lower()
    try:
        encoded_str = "".join(_LETTERS2[ch] for ch in s)
    except KeyError as e:
        raise ValueError("letters2 encoding supports only A-Z letters") from e
    return encoded_str.encode('ascii')

def encode_text(text: str, encoding: str) -> bytes:
    if encoding == 'utf8ddd':
        return text_to_utf8_digits(text)
    elif encoding == 'letters2':
        return text_to_letters2_digits(text)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

def ensure_cache(digits: int, cache_path: str, source_url: str, show_progress: bool = False) -> None:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) >= digits:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    written = 0
    tmp_path = cache_path + ".part"
    with urlopen(source_url, timeout=60) as resp, open(tmp_path, "wb") as out:
        CHUNK = 8 * 1024 * 1024
        last_pct = -1
        while written < digits:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            chunk = chunk.translate(IDENTITY, NON_DIGITS)
            if not chunk:
                continue
            need = digits - written
            to_write = chunk[:need]
            out.write(to_write)
            written += len(to_write)
            if show_progress:
                pct = int(written * 100 / digits)
                if pct != last_pct:
                    print(f"\rDownloading π digits: {written/1e6:.1f}/{digits/1e6:.1f} MB ({pct}%)", end="", flush=True)
                    last_pct = pct
        if show_progress and last_pct >= 0:
            print()
    if written < digits:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise RuntimeError(f"Source did not provide enough digits (got {written}, need {digits}).")
    os.replace(tmp_path, cache_path)

def find_in_pi_bytes(needle: bytes, digits: int, cache_path: str, source_url: str, show_progress: bool = False) -> int:
    ensure_cache(digits, cache_path, source_url, show_progress=show_progress)
    if not show_progress:
        with open(cache_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return mm.find(needle, 0, digits)
    # Progress-enabled search: chunked mm.find with overlap
    overlap = max(0, len(needle) - 1)
    CHUNK = 128 * 1024 * 1024
    with open(cache_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        last_pct = -1
        start = 0
        while start < digits:
            end = min(digits, start + CHUNK)
            search_start = max(0, start - overlap)
            pos = mm.find(needle, search_start, end)
            if pos != -1:
                return pos
            if show_progress:
                pct = int(end * 100 / digits)
                if pct != last_pct:
                    print(f"\rSearching: {end/1e6:.1f}/{digits/1e6:.1f} MB ({pct}%)", end="", flush=True)
                    last_pct = pct
            start = end
        if show_progress and last_pct >= 0:
            print()
    return -1

def approx_prob_found(needle_len: int, N: int, alphabet: int = 10) -> float:
    """
    Poisson approximation assuming IID uniform digits:
      λ ≈ (N - m + 1) / alphabet^m,  P(found ≥1) ≈ 1 - exp(-λ)
    Accurate when λ is small (rare pattern) and good in practice for our use.
    """
    if N < needle_len:
        return 0.0
    lam = (N - needle_len + 1) / (alphabet ** needle_len)
    # guard extreme cases
    if lam > 50:  # avoid underflow in exp(-lam)
        return 1.0
    return 1.0 - math.exp(-lam)


# ---- Multi-word scan (4-letter words) with progress ----
def _encode_bytes_for_automaton(s: str, encoding: str) -> bytes:
    return encode_text(s, encoding)

def load_words_of_length(path: str, wordlen: int, limit: int = 10000) -> List[str]:
    words: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if len(w) == wordlen and w.isascii() and w.isalpha():
                words.append(w)
                if len(words) >= limit:
                    break
    return words

def build_automaton(words: List[str], encoding: str):
    try:
        import ahocorasick  # pyahocorasick package
    except Exception as e:
        raise RuntimeError("Requires 'pyahocorasick' package: pip install pyahocorasick") from e

    A = ahocorasick.Automaton()
    meta: Dict[int, Tuple[str, int]] = {}
    idx = 0
    for w in words:
        pat_bytes = _encode_bytes_for_automaton(w, encoding)
        pat_str = pat_bytes.decode('ascii')
        A.add_word(pat_str, idx)
        meta[idx] = (w, len(pat_str))
        idx += 1
    A.make_automaton()
    max_pat_len = max((l for _, l in meta.values()), default=0)
    return A, meta, max_pat_len

def scan_pi_for_words(pi_path: str, digits: int, A, meta: Dict[int, Tuple[str, int]], max_pat_len: int, show_progress: bool = False) -> Dict[str, int]:
    CHUNK = 16 * 1024 * 1024
    overlap = max(0, max_pat_len - 1)
    found: Dict[str, int] = {}
    consumed = 0
    tail = b""
    total = digits
    last_pct = -1

    with open(pi_path, "rb") as f:
        while consumed < total:
            to_read = min(CHUNK, total - consumed)
            chunk = f.read(to_read)
            if not chunk:
                break

            buf = tail + chunk
            text = buf.decode('ascii', errors='ignore')
            base_offset = consumed - len(tail)

            for end_pos, idx in A.iter(text):
                word, pat_len = meta[idx]
                if word in found:
                    continue
                abs_pos = base_offset + end_pos - pat_len + 1
                if abs_pos < 0 or abs_pos >= digits:
                    continue
                found[word] = abs_pos
                if len(found) == len(meta):
                    if show_progress and last_pct >= 0:
                        print()
                    return found

            # keep tail for overlap across chunks
            if len(buf) >= overlap:
                tail = buf[-overlap:]
            else:
                tail = buf

            consumed += len(chunk)
            if show_progress:
                pct = int(consumed * 100 / total)
                if pct != last_pct:
                    print(f"\rScanning words: {consumed/1e6:.1f}/{total/1e6:.1f} MB ({pct}%)  found={len(found)}", end="", flush=True)
                    last_pct = pct

    if show_progress and last_pct >= 0:
        print()
    return found

def ensure_wordlist(cache_path: str, source_url: str = DEFAULT_WORDLIST_URL, show_progress: bool = False) -> str:
    """Download a public English word list if not present. Returns the local path."""
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    written = 0
    tmp_path = cache_path + ".part"
    with urlopen(source_url, timeout=60) as resp, open(tmp_path, "wb") as out:
        total = 0
        try:
            total = int(resp.headers.get("Content-Length", "0"))
        except Exception:
            total = 0
        CHUNK = 8 * 1024 * 1024
        last_pct = -1
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if show_progress and total > 0:
                pct = int(written * 100 / total)
                if pct != last_pct:
                    print(f"\rDownloading wordlist: {written/1e6:.1f}/{total/1e6:.1f} MB ({pct}%)", end="", flush=True)
                    last_pct = pct
        if show_progress and last_pct >= 0:
            print()
    if written == 0:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise RuntimeError("Failed to download wordlist.")
    os.replace(tmp_path, cache_path)
    return cache_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="String to search (encoding depends on --encoding).")
    parser.add_argument("--digits", type=int, required=True, help="How many π digits to search.")
    parser.add_argument("--cache", type=str, default=CACHE_FILE, help="Digits-only cache file path.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="URL to stream π digits (digits-only after filtering).")
    parser.add_argument("--progress", action="store_true", help="Show progress during download and search (slightly slower search).")
    parser.add_argument("--wordlist", type=str, default=None, help="Path to word list (one word per line). If set, scans all 4-letter words. If omitted, a default list will be downloaded.")
    parser.add_argument("--limit", type=int, default=10000, help="Max words to load from wordlist.")
    parser.add_argument("--wordlen", type=int, default=4, help="Word length to filter from wordlist when scanning.")
    parser.add_argument("--encoding", type=str, default="utf8ddd", choices=["utf8ddd", "letters2"], help="Encoding for text/words: utf8ddd (bytes→000-255), letters2 (A-Z→01..26)")
    parser.add_argument("--words-only", action="store_true", help="In wordlist mode, print only matching words (one per line), no indices or summary.")
    args = parser.parse_args()

    if args.wordlist is not None:
        # Multi-word scan path
        ensure_cache(args.digits, args.cache, args.source, show_progress=args.progress)
        wordlist_path = args.wordlist or ensure_wordlist(DEFAULT_WORDLIST_CACHE, show_progress=args.progress)
        words = load_words_of_length(wordlist_path, wordlen=args.wordlen, limit=args.limit)
        A, meta, max_pat_len = build_automaton(words, encoding=args.encoding)
        found = scan_pi_for_words(args.cache, args.digits, A, meta, max_pat_len, show_progress=args.progress)
        if args.words_only:
            for w, pos in sorted(found.items(), key=lambda x: x[1]):
                print(w)
        else:
            # Print results: position\tword (sorted by position)
            for w, pos in sorted(found.items(), key=lambda x: x[1]):
                print(f"{pos}\t{w}")
            # Summary line to stderr-like (just print at end)
            print(f"FOUND {len(found)}/{len(words)} words")
    else:
        # Single-string search path (requires --target)
        if not args.target:
            raise SystemExit("--target is required when --wordlist is not provided")
        needle = encode_text(args.target, args.encoding)
        # Output the digits we will search for
        print(needle.decode("ascii"))

        pos = find_in_pi_bytes(needle, args.digits, args.cache, args.source, show_progress=args.progress)
        print(pos)

        # Probability (random-digit model)
        p = approx_prob_found(len(needle), args.digits, alphabet=10)
        print(f"{p:.8g}")