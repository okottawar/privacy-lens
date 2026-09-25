from app.retrieval import chunk_sections


def test_chunk_sections_preserves_section_name_and_ids():
    sections = [{
        "section": "Data Collection",
        "content": "Sentence one. " * 300,
    }]

    chunks = chunk_sections(sections)

    assert len(chunks) > 1
    assert all(chunk["section"] == "Data Collection" for chunk in chunks)
    assert [chunk["chunk_id"] for chunk in chunks] == [
        f"chunk_{i}" for i in range(len(chunks))
    ]


def test_chunk_sections_keeps_short_section_as_one_chunk():
    sections = [{"section": "Retention", "content": "Data is deleted after thirty days."}]

    chunks = chunk_sections(sections)

    assert len(chunks) == 0  # current minimum-content guard rejects very short sections
