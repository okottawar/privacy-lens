from app.config import get_settings


def test_default_embedding_configuration():
    settings = get_settings()

    assert settings.embedding_provider == "nvidia"
    assert settings.embedding_model == "nvidia/nemotron-3-embed-1b"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_concurrency == 4


def test_default_retrieval_weights():
    settings = get_settings()

    assert settings.chat_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert settings.reasoning_concurrency == 2
    assert settings.reasoning_timeout_seconds == 30.0
    assert settings.reasoning_evidence_chunks == 4
    assert settings.reasoning_chunk_chars == 900
    assert settings.retrieval_dense_weight == 0.75
    assert settings.retrieval_lexical_weight == 0.25


def test_default_allowed_origins_keeps_public_demo_compatible():
    settings = get_settings()

    assert settings.allowed_origins == ["*"]
