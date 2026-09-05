import hashlib
import sys
from _hashlib import HASH
from pathlib import Path

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.config import ensense_pin  # noqa: E402

_ENSENSE_DIR = _ROOT / "ensense"


def _tree_hash(root: Path) -> str:
    hasher: HASH = hashlib.sha256()

    for path in sorted(root.rglob("*")):
        if path.is_file():
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(path.read_bytes())

    return hasher.hexdigest()


def test_ensense_pin_file_exists_and_is_a_sha() -> None:
    pin: str = ensense_pin()
    assert len(pin) == 40, f"expected a 40-char git SHA, got {pin!r}"
    int(pin, 16)


def test_ensense_vendored_and_present() -> None:
    assert _ENSENSE_DIR.is_dir(), "ensense/ is missing -- see docs/ensense_interface.md"
    assert (_ENSENSE_DIR / "LICENSE").is_file()
    assert (_ENSENSE_DIR / "src" / "sensitive.py").is_file()
    assert not (_ENSENSE_DIR / ".git").exists(), (
        "ensense/ must be vendored as plain files, not a nested git repo"
    )


def test_ensense_tree_hash_is_stable_across_two_computations() -> None:
    assert _tree_hash(_ENSENSE_DIR) == _tree_hash(_ENSENSE_DIR)
