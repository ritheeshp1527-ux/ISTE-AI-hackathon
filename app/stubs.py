"""Stubs for retrieval and answer generation."""

import uuid
from app.contracts import Answer, Chunk, Citation, Passage


FAKE_CHUNKS = [
    Chunk(
        chunk_id="SOP-0042-v2-p03-c01",
        doc_id="SOP-0042-v2",
        title="Sample Handling Procedure",
        page=3,
        section="Storage and Handling",
        access_group="quality",
        text="All temperature-sensitive reagents and biological samples must Store at 2-8 °C in a verified cold storage unit.",
    ),
    Chunk(
        chunk_id="SOP-0010-v1-p01-c01",
        doc_id="SOP-0010-v1",
        title="General Laboratory Safety",
        page=1,
        section="Personal Protective Equipment",
        access_group="general",
        text="Personnel must wear approved safety glasses, lab coats, and nitrile gloves in all operational areas.",
    ),
    Chunk(
        chunk_id="SOP-0005-v3-p02-c01",
        doc_id="SOP-0005-v3",
        title="Standard Equipment Operation Guidelines",
        page=2,
        section="Calibration and Setup",
        access_group="operations",
        text="Inspect instrumentation and verify calibration logs prior to initiating any production or test run.",
    ),
]


def get_fake_passages() -> list[Passage]:
    scores = [0.95, 0.85, 0.75]
    return [
        Passage(chunk=chunk, score=score)
        for chunk, score in zip(FAKE_CHUNKS, scores)
    ]


def stub_retrieve(
    question: str,
    user_groups: list[str],
    k: int = 5,
    force_low_score: bool = False,
) -> list[Passage]:
    """Retrieve fake passages filtered by the user's authorized access groups."""
    all_passages = get_fake_passages()
    filtered = [p for p in all_passages if p.chunk.access_group in user_groups]
    if force_low_score:
        filtered = [
            Passage(chunk=p.chunk, score=0.1)
            for p in filtered
        ]
    return filtered[:k]


def stub_answer(question: str, passages: list[Passage]) -> Answer:
    """Return a fixed Answer with one citation for the first passage and a random request_id."""
    citations: list[Citation] = []
    if passages:
        first = passages[0]
        citations.append(
            Citation(
                source_id=first.chunk.doc_id,
                title=first.chunk.title,
                page=first.chunk.page,
                section=first.chunk.section,
                chunk_id=first.chunk.chunk_id,
            )
        )

    return Answer(
        answer="According to standard operating procedures, samples must be handled and stored in compliance with temperature and safety controls.",
        citations=citations,
        status="answered",
        request_id=str(uuid.uuid4()),
    )
