import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_one
from src.config import PRED_DIR, RESULTS_DIR, ensure_dirs
from src.data import load_labels

DEV = ["P-1", "E-2", "A-2", "T-1", "D-1", "M-1", "M-6", "P-10", "F-7", "C-1"]

ensure_dirs()
chans = sorted(set(load_labels()["chan_id"]) - set(DEV))
todo = [c for c in chans if not (PRED_DIR / f"{c}.npz").exists()]
print(f"{len(chans)} final channels, {len(todo)} still to train")

t0 = time.time()
for i, c in enumerate(todo, 1):
    t = time.time()
    try:
        train_one.main(c, 35, 5)
    except Exception:
        with open(RESULTS_DIR / "train_errors.txt", "a") as f:
            f.write(f"{c}\n{traceback.format_exc()}\n")
        print("FAILED", c)
    el = time.time() - t0
    print(f"=== {i}/{len(todo)} {c}: {time.time() - t:.0f}s, "
          f"elapsed {el / 60:.0f} min, ETA {el / i * (len(todo) - i) / 60:.0f} min")