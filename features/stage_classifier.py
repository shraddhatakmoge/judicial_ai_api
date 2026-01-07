def classify_stage(purpose: str) -> str:
    if not purpose:
        return "Other"

    text = str(purpose).lower()

    if "admission" in text:
        return "Admission"

    if "evidence" in text or "pw" in text:
        return "Evidence"

    if "argument" in text or "heard" in text:
        return "Arguments"

    if "judgment" in text or "order" in text or "disposed" in text:
        return "Judgment"

    if "adjourn" in text or "posted" in text:
        return "Adjournment"

    return "Other"


def classify_stage(purpose: str) -> str:
    if not purpose:
        return "Other"

    text = str(purpose).lower()

    if "admission" in text:
        return "Admission"
    if "evidence" in text or "pw" in text:
        return "Evidence"
    if "argument" in text or "heard" in text:
        return "Arguments"
    if "judgment" in text or "order" in text or "disposed" in text:
        return "Judgment"
    if "adjourn" in text or "posted" in text:
        return "Adjournment"

    return "Other"
