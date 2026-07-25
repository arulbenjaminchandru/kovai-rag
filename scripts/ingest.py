"""Build the ChromaDB index from the Kovai Finserv policy handbook.

One chunk per policy section, split on "## " markdown headings. The index is
deleted and rebuilt from scratch on every run, so a policy section removed from
the source document really disappears from the index.

Run as:  python scripts/ingest.py
"""

import re
import shutil
import sys
from pathlib import Path

# This script lives in scripts/, so the project root must be on sys.path
# before we can import app.config.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402

from app.config import settings  # noqa: E402

POLICY_FILE = PROJECT_ROOT / "data" / "kovai_policies.md"


def split_into_sections(text):
    """Split markdown into one chunk per "## " section.

    Anything that does not start with "## " is skipped, so the document title
    block at the top of the file is not indexed as a chunk.
    """
    sections = []
    for part in re.split(r"\n(?=## )", text):
        part = part.strip()
        if not part.startswith("## "):
            continue
        # The heading text is the first line, minus the "## " marker.
        title = part.split("\n", 1)[0][len("## "):].strip()
        sections.append({"title": title, "text": part})
    return sections


def main():
    text = POLICY_FILE.read_text(encoding="utf-8")
    sections = split_into_sections(text)

    if not sections:
        raise SystemExit(f"No '## ' sections found in {POLICY_FILE}")

    # Rebuild from scratch so removed policy sections really disappear.
    chroma_path = Path(settings.chroma_path)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.create_collection(
        name=settings.collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )

    collection.add(
        ids=[f"section-{i}" for i in range(len(sections))],
        documents=[s["text"] for s in sections],
        metadatas=[
            {"title": s["title"], "source": POLICY_FILE.name} for s in sections
        ],
    )

    print(f"Indexed {len(sections)} chunks into {chroma_path}")
    print(f"Collection: {settings.collection_name}")
    for i, section in enumerate(sections, start=1):
        print(f"  {i}. {section['title']}")


if __name__ == "__main__":
    main()
