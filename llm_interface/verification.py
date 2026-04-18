"""Deterministic verification pipeline for triplet extraction outputs.

Four-stage verification that checks each extracted field against the source
report text, classifying failures as hallucinated or paraphrased.
"""
import re
from enum import Enum


class VerificationStage(Enum):
    EXACT = 1
    NORMALISED = 2
    TOKEN_OVERLAP = 3
    FAILED = 4


class FailureType(Enum):
    HALLUCINATED = "hallucinated"
    PARAPHRASED = "paraphrased"


ABBREVIATION_MAP = {
    "rul": "right upper lobe",
    "rml": "right middle lobe",
    "rll": "right lower lobe",
    "lul": "left upper lobe",
    "lll": "left lower lobe",
    "bil": "bilateral",
    "lat": "lateral",
    "med": "medial",
    "ant": "anterior",
    "post": "posterior",
    "sup": "superior",
    "inf": "inferior",
}


def normalise_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    for abbr, expansion in ABBREVIATION_MAP.items():
        text = re.sub(r'\b' + abbr + r'\b', expansion, text)
    return text


def exact_substring_match(field_value, report_text):
    return field_value.lower() in report_text.lower()


def normalised_match(field_value, report_text):
    return normalise_text(field_value) in normalise_text(report_text)


def token_overlap_match(field_value, report_text, threshold=0.8):
    field_tokens = set(normalise_text(field_value).split())
    report_tokens = set(normalise_text(report_text).split())
    if not field_tokens:
        return False, 0.0
    overlap = len(field_tokens & report_tokens) / len(field_tokens)
    return overlap >= threshold, overlap


def classify_failure(field_value, report_text):
    field_tokens = set(normalise_text(field_value).split())
    report_tokens = set(normalise_text(report_text).split())
    overlap = len(field_tokens & report_tokens)
    if overlap == 0:
        return FailureType.HALLUCINATED
    return FailureType.PARAPHRASED


def verify_field(field_value, report_text):
    if not field_value or field_value.lower() == "none":
        return VerificationStage.EXACT, None

    if exact_substring_match(field_value, report_text):
        return VerificationStage.EXACT, None

    if normalised_match(field_value, report_text):
        return VerificationStage.NORMALISED, None

    passed, overlap = token_overlap_match(field_value, report_text)
    if passed:
        return VerificationStage.TOKEN_OVERLAP, None

    failure_type = classify_failure(field_value, report_text)
    return VerificationStage.FAILED, failure_type


def verify_triplet(triplet, report_text):
    results = {}
    for field in ["observation", "location", "properties"]:
        value = triplet.get(field, "")
        stage, failure = verify_field(value, report_text)
        results[field] = {"stage": stage, "failure": failure, "value": value}
    return results


class VerificationAction(Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    REJECT = "reject"


def decide_action(verification_results):
    stages = [r["stage"] for r in verification_results.values()]
    if any(s == VerificationStage.FAILED for s in stages):
        return VerificationAction.REJECT
    if any(s == VerificationStage.TOKEN_OVERLAP for s in stages):
        return VerificationAction.FLAG
    return VerificationAction.ACCEPT


def verify_extraction(triplets, report_text):
    accepted = []
    flagged = []
    rejected = []

    for triplet in triplets:
        results = verify_triplet(triplet, report_text)
        action = decide_action(results)

        entry = {
            "triplet": triplet,
            "verification": results,
            "action": action,
        }

        if action == VerificationAction.ACCEPT:
            accepted.append(entry)
        elif action == VerificationAction.FLAG:
            flagged.append(entry)
        else:
            rejected.append(entry)

    return {
        "accepted": accepted,
        "flagged": flagged,
        "rejected": rejected,
        "summary": {
            "total": len(triplets),
            "accepted": len(accepted),
            "flagged": len(flagged),
            "rejected": len(rejected),
        },
    }


def build_reextraction_prompt(rejected_entry, report_text):
    triplet = rejected_entry["triplet"]
    verification = rejected_entry["verification"]

    failed_fields = [
        f for f, r in verification.items()
        if r["stage"] == VerificationStage.FAILED
    ]
    failure_types = [
        verification[f]["failure"].value for f in failed_fields
    ]

    import json
    return (
        f"The following extraction was rejected because the "
        f"{', '.join(failed_fields)} field(s) could not be verified against "
        f"the source report.\n\n"
        f"ORIGINAL EXTRACTION:\n{json.dumps(triplet, indent=2)}\n\n"
        f"FAILURE TYPE: {', '.join(failure_types)}\n\n"
        f"REPORT TEXT:\n{report_text}\n\n"
        f"Please re-extract this finding. You MUST use ONLY words and phrases "
        f"that appear verbatim in the report text above. Do not paraphrase, "
        f"synonym-substitute, or infer terms that are not explicitly written.\n\n"
        f"If the finding cannot be expressed using only text from the report, "
        f"respond with \"CANNOT_VERIFY\".\n\n"
        f"OUTPUT FORMAT:\n"
        f"{{\"observation\": \"...\", \"location\": \"...\", \"properties\": \"...\"}}"
    )