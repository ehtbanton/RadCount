"""Embedding-based index correction for findings extraction.

Uses Sentence-BERT (all-MiniLM-L6-v2) to correct misaligned word indices
returned by the LLM during extraction.
"""
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def compute_cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def correct_index(extracted_word, predicted_index, indexed_report, window=10):
    """Correct a misaligned index using embedding similarity.

    Args:
        extracted_word: The word the LLM intended to extract.
        predicted_index: The index the LLM returned.
        indexed_report: List of [word, index] pairs from the report.
        window: Half-width of the search window around predicted_index.

    Returns:
        dict with corrected_index, similarity, and similarity_profile.
    """
    model = _get_model()

    index_to_word = {idx: word for word, idx in indexed_report}

    window_start = max(0, predicted_index - window)
    window_end = predicted_index + window
    candidate_indices = [
        idx for idx in range(window_start, window_end + 1)
        if idx in index_to_word
    ]

    if not candidate_indices:
        return {
            "corrected_index": predicted_index,
            "similarity": 0.0,
            "exact_match": False,
        }

    target_embedding = model.encode([extracted_word])[0]
    candidate_words = [index_to_word[idx] for idx in candidate_indices]
    candidate_embeddings = model.encode(candidate_words)

    similarities = [
        compute_cosine_similarity(target_embedding, ce)
        for ce in candidate_embeddings
    ]

    best_pos = int(np.argmax(similarities))
    corrected_index = candidate_indices[best_pos]
    best_similarity = similarities[best_pos]

    exact_match = index_to_word.get(corrected_index, "").lower() == extracted_word.lower()

    return {
        "corrected_index": corrected_index,
        "similarity": best_similarity,
        "exact_match": exact_match,
        "original_index": predicted_index,
        "corrected_word": index_to_word.get(corrected_index, ""),
        "profile": {
            candidate_indices[i]: similarities[i]
            for i in range(len(candidate_indices))
        },
    }


def validate_and_correct_extractions(extractions, indexed_report, threshold=0.7):
    """Validate and correct indices for a list of extractions.

    Each extraction should have an 'observation' field with [word, index].

    Returns:
        List of corrected extractions and a summary of corrections made.
    """
    index_to_word = {idx: word for word, idx in indexed_report}
    corrected = []
    stats = {"total": 0, "correct": 0, "corrected": 0, "failed": 0}

    for extraction in extractions:
        stats["total"] += 1
        observation = extraction.get("observation", [])
        if not observation or len(observation) != 2:
            stats["failed"] += 1
            continue

        word, idx = observation
        actual_word = index_to_word.get(idx, "")

        if actual_word.lower() == word.lower():
            stats["correct"] += 1
            corrected.append(extraction)
            continue

        result = correct_index(word, idx, indexed_report)

        if result["similarity"] >= threshold:
            stats["corrected"] += 1
            new_extraction = dict(extraction)
            new_extraction["observation"] = [result["corrected_word"], result["corrected_index"]]
            corrected.append(new_extraction)
        else:
            stats["failed"] += 1

    return corrected, stats