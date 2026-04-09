"""Download 12 synthetic face avatars from a HuggingFace dataset.

Pulls a small fixed sample from `javi22/this-person-does-not-exist-10k`
on HuggingFace and saves them under
`Benchmark_Pipeline/data/image/Chat_Memory_Test/faces/` named by persona id.

Source dataset
--------------
- Name:    javi22/this-person-does-not-exist-10k
- URL:     https://huggingface.co/datasets/javi22/this-person-does-not-exist-10k
- Author:  Javier (HuggingFace user `javi22`)
- Format:  10,000 individual JPG files (1024x1024 px)
- License: MIT (commercial + research use, attribution required)
- Content: Wholly synthetic AI-generated faces (StyleGAN-family). The
           images do NOT depict any real person — they are GAN samples.

We download exactly 12 faces by fixed indices so the selection is
reproducible. We do not bundle or redistribute the dataset; this script
fetches directly from HuggingFace at run time. The full attribution
notice (`FACE_DATA_NOTICE.md`) is shipped from the tracked source
location and copied next to the downloaded faces so anyone who has
the data sees the licensing info.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Strip token vars that the env might inject (mirrors CLAUDE.md guidance)
for k in list(os.environ):
    if "TOKEN" in k or "HUGGING" in k:
        del os.environ[k]

from huggingface_hub import hf_hub_download  # noqa: E402


REPO = "javi22/this-person-does-not-exist-10k"
DEST = Path(__file__).resolve().parents[2] / "Benchmark_Pipeline" / "data" / "image" / "Chat_Memory_Test" / "faces"


PERSONAS = [
    # Names assigned after eyeballing the actual downloaded faces so the
    # apparent age / gender of each photo matches its narrative role.
    ("P01", "marcus"),  # BOSS network: manager (older man, glasses)
    ("P02", "priya"),   # BOSS + ROOM crossover (woman 30s)
    ("P03", "daniel"),  # BOSS + COL crossover (young man, new hire)
    ("P04", "tomas"),   # FAM: teenage son (androgynous young face, curly hair)
    ("P05", "helen"),   # FAM: mother (woman 30s)
    ("P06", "jordan"),  # TRI + COL crossover (woman 30s)
    ("P07", "sara"),    # ROOM: Priya's roommate (young blonde woman)
    ("P08", "owen"),    # distractor pool (man 40s with glasses)
    ("P09", "mia"),     # FAM: little sister (child girl ~8)
    ("P10", "ryan"),    # TRI vertex (man 30s)
    ("P11", "kai"),     # distractor pool (gym friend, hat/beard)
    ("P12", "elena"),   # TRI vertex (young woman 20s)
]

# Fixed indices spaced across the 10k pool to get visual variety.
# Using fixed indices guarantees reproducibility.
INDICES = [137, 412, 855, 1023, 1487, 2210, 3045, 3712, 4521, 5304, 6118, 7203]


def _copy_notice() -> None:
    """Copy FACE_DATA_NOTICE.md next to the downloaded faces.

    The data directory is gitignored, so the notice is shipped from the
    tracked Image_Generator/chatUI/ source location and re-copied at fetch
    time so anyone with the data sees it.
    """
    src = Path(__file__).resolve().parent / "FACE_DATA_NOTICE.md"
    dst = DEST.parent / "FACE_DATA_NOTICE.md"
    if src.exists():
        shutil.copy(src, dst)
        print(f"  notice:  {dst}")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {DEST}")
    _copy_notice()

    for (pid, name), idx in zip(PERSONAS, INDICES):
        src_name = f"face ({idx}).jpg"
        dst = DEST / f"{pid}_{name}.jpg"
        if dst.exists():
            print(f"  exists:  {dst.name}")
            continue
        try:
            cached = hf_hub_download(
                repo_id=REPO,
                filename=src_name,
                repo_type="dataset",
            )
        except Exception as exc:
            print(f"  FAILED  {dst.name}: {exc}")
            continue
        shutil.copy(cached, dst)
        print(f"  saved:   {dst.name}  ({Path(cached).stat().st_size} bytes)")

    print("\nFinal directory:")
    for f in sorted(DEST.glob("*.jpg")):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
