from __future__ import annotations


def detect_sections(
    speaker_types: list[str],
    paragraph_indices: list[int],
) -> list[str]:
    """Label each row as 'prepared_remarks' or 'q_and_a'.

    The boundary is the first paragraph where speaker_type == 'analyst'.
    Everything before that paragraph is prepared remarks.
    If no analyst is found, all rows are labeled 'unknown'.

    Parameters
    ----------
    speaker_types : per-row speaker type ('management', 'analyst', etc.)
    paragraph_indices : per-row paragraph index from the transcript

    Returns
    -------
    List of section labels, same length as inputs.
    """
    if not speaker_types:
        return []

    # Find the paragraph index where the first analyst appears
    boundary_para: int | None = None
    for stype, pidx in zip(speaker_types, paragraph_indices):
        if stype == "analyst":
            boundary_para = pidx
            break

    if boundary_para is None:
        return ["unknown"] * len(speaker_types)

    return [
        "prepared_remarks" if pidx < boundary_para else "q_and_a"
        for pidx in paragraph_indices
    ]
