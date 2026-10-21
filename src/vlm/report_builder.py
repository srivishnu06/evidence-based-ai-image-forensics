
def build_forensic_report(cv_result, vlm_observation=None):
    """
    Combine independent CV predictions and an optional
    VLM observation into a cautious forensic report.

    The VLM never determines the authenticity verdict.
    """

    probability = float(cv_result["ai_probability"])
    threshold = float(cv_result["auth_threshold"])

    verdict = (
        "AI-generated"
        if probability >= threshold
        else "Authentic"
    )

    category = cv_result["strongest_artifact"]
    artifact_score = float(
        cv_result["strongest_artifact_probability"]
    )

    flagged = bool(
        cv_result["artifact_detected"].get(
            category, False
        )
    )

    region_count = len(cv_result["regions"])

    evidence_available = (
        flagged and region_count > 0
    )

    observation = (
        str(vlm_observation).strip()
        if vlm_observation is not None
        else ""
    )

    sections = []

    sections.append(
        "CV AUTHENTICITY PREDICTION\n"
        f"{verdict} "
        f"(AI probability: {probability:.4f}; "
        f"threshold: {threshold:.4f}). "
        "This is a model prediction, not "
        "independent verification of image origin."
    )

    sections.append(
        "STAGE 2 ARTIFACT SIGNAL\n"
        f"Highest-scoring category: {category}. "
        f"Category probability: {artifact_score:.4f}. "
        f"Category flagged: {flagged}. "
        f"Candidate evidence regions: {region_count}."
    )

    if evidence_available:
        if observation:
            sections.append(
                "VLM VISUAL OBSERVATION "
                "(UNVERIFIED)\n"
                + observation
            )
        else:
            sections.append(
                "VLM VISUAL OBSERVATION\n"
                "No visual observation is available."
            )
    else:
        sections.append(
            "LOCALISATION WARNING\n"
            "The highest-scoring artifact does not "
            "have sufficient threshold-supported "
            "localised evidence. No VLM observation "
            "is presented as supporting evidence."
        )

    limitations = (
        "EVIDENCE LIMITATIONS\n"
        "Artifact scores and highlighted regions are "
        "model estimates, not verified defects. "
        "Any VLM observation requires visual review. "
        "Neither an artifact prediction nor its "
        "explanation establishes image origin."
    )

    if verdict == "Authentic" and flagged:
        limitations += (
            " Stage 1 predicts authentic even though "
            "Stage 2 flags a possible inconsistency. "
            "These are separate model outputs and "
            "should not be treated as a confirmed "
            "authenticity contradiction."
        )

    sections.append(limitations)

    return {
        "verdict": verdict,
        "ai_probability": probability,
        "artifact_category": category,
        "artifact_probability": artifact_score,
        "artifact_flagged": flagged,
        "evidence_regions": region_count,
        "evidence_available": evidence_available,
        "vlm_observation_used": bool(
            evidence_available and observation
        ),
        "report": "\n\n".join(sections),
    }
