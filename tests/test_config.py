from app.config import get_settings


def test_default_embedding_configuration():
    settings = get_settings()

    assert settings.embedding_provider == "nvidia"
    assert settings.embedding_model == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_concurrency == 4


def test_default_retrieval_weights():
    settings = get_settings()

    assert settings.retrieval_dense_weight == 0.75
    assert settings.retrieval_lexical_weight == 0.25
