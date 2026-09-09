from lucia.embeddings import cosine_similarity, normalize_text, text_vector


def test_normalize_text_removes_accents_and_case():
    assert normalize_text("Lucía LOCAL") == "lucia local"


def test_cosine_similarity_matches_related_tokens():
    left = text_vector("Lucía local memory")
    right = text_vector("local memory system")
    unrelated = text_vector("avatar animation")

    assert cosine_similarity(left, right) > cosine_similarity(left, unrelated)
    assert cosine_similarity(left, left) == 1.0
