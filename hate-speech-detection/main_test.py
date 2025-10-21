"""
Demo script that uses ExplainableTweetClassifier and the helper functions:
- moderate_content
- generate_appeal_response
- analyze_false_positives
"""

from typing import Dict, Any, List
from inference.explainable_classifier import ExplainableTweetClassifier


# Initialize the classifier (assumes model is already trained and weights available)
classifier = ExplainableTweetClassifier()


def moderate_content(text: str) -> Dict[str, Any]:
    """
    Run the classifier and return a compact moderation decision dict.

    Uses include_severity=True so the classifier returns 'severity' info.
    Handles missing keys gracefully.
    """
    try:
        result = classifier.classify_with_explanation(text, include_severity=True, verbose=False)
    except Exception as e:
        return {"error": str(e), "text": text}

    # Defensive extraction of fields that may or may not be present
    cls = result.get("class") or result.get("prediction") or result.get("label")
    explanation = result.get("explanation", {})
    # Try to find the keywords explanation (fall back to entire explanation)
    kw_expl = explanation.get("explanations", {}).get("keywords", {}).get("explanation") \
              or explanation.get("explanations", {}).get("lime", {}).get("explanation") \
              or str(explanation)

    severity = (result.get("severity") or {}).get("severity_label") if isinstance(result.get("severity"), dict) else result.get("severity")
    action = (result.get("action") or {}).get("primary_action") if isinstance(result.get("action"), dict) else result.get("action")

    return {
        "input": text,
        "should_remove": cls in [0, 1] if cls is not None else False,
        "class": cls,
        "reason": kw_expl,
        "severity": severity,
        "action": action,
        "raw": result
    }


def generate_appeal_response(text: str) -> str:
    """
    Generate a human-readable appeal response using LIME explanations.
    If LIME is unavailable in the classifier methods, falls back to keywords explanation.
    """
    try:
        result = classifier.classify_with_explanation(text, methods=["lime"], verbose=False)
    except Exception:
        # Fallback to keywords explanation if lime isn't available or fails
        result = classifier.classify_with_explanation(text, methods=["keywords"], verbose=False)

    prediction = result.get("prediction") or result.get("class") or result.get("label") or "unknown"
    confidence = result.get("confidence") or result.get("score") or 0.0

    # safe extraction of LIME positive contributions
    lime_expl = result.get("explanation", {}).get("explanations", {}).get("lime", {})
    pos_contrib = lime_expl.get("positive_contributions", []) if isinstance(lime_expl, dict) else []

    # fallback to keywords explanation if no lime contributions
    if not pos_contrib:
        kw = result.get("explanation", {}).get("explanations", {}).get("keywords", {}).get("explanation", "No keyword explanation available")
        top_words = kw
    else:
        top_words = ", ".join([w for w, _ in pos_contrib[:5]])

    detected_patterns = result.get("explanation", {}).get("explanations", {}).get("keywords", {}).get("explanation", "N/A")

    return f"""Your content was flagged as: {prediction}

1. Classification confidence: {confidence:.1%}
2. Key contributing words: {top_words}
3. Detected harmful patterns: {detected_patterns}

If you believe this is a mistake, please contact support and provide this message.
"""


def analyze_false_positives(texts: List[str]) -> None:
    """
    Iterate a list of likely false-positive texts and print classifier outputs
    (suitable for building a review dataset).
    """
    for text in texts:
        try:
            result = classifier.classify_with_explanation(text, methods=["lime", "keywords"], verbose=False)
        except Exception as e:
            print(f"Error classifying '{text}': {e}")
            continue

        prediction = result.get("prediction") or result.get("class") or result.get("label")
        confidence = result.get("confidence") or result.get("score", None)
        lime_expl = result.get("explanation", {}).get("explanations", {}).get("lime", {})
        lime_summary = lime_expl.get("explanation") if isinstance(lime_expl, dict) else str(lime_expl)

        print("=== SAMPLE ===")
        print(f"Text: {text}")
        print(f"Predicted label: {prediction} (confidence: {confidence})")
        print("LIME explanation (summary):")
        print(lime_summary)
        print()


def batch_moderate(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Example showing how to moderate a batch of texts sequentially.
    If the classifier supports a batch mode, you can replace the loop with that call.
    """
    results = []
    for t in texts:
        res = moderate_content(t)
        results.append(res)
    return results


if __name__ == "__main__":
    # Example inputs demonstrating violent/hateful content, borderline cases, and benign text.
    examples = [
        "I will kill you fucking bitch you worthless trash...",
        "You're a fucking idiot",
        "I hate when my code doesn't work",
        "This movie was terrible",
        "I strongly disagree with this policy",
        "Thanks for your help — I appreciate it!"
    ]

    # 1) Run moderation on the violent example and print a prettified result
    print("---- Single moderation (violent example) ----")
    mod_result = moderate_content(examples[0])
    # Nicely print key fields
    print(f"Input: {mod_result['input']}")
    print(f"Should remove: {mod_result['should_remove']}")
    print(f"Predicted class: {mod_result.get('class')}")
    print(f"Severity: {mod_result.get('severity')}")
    print(f"Action: {mod_result.get('action')}")
    print(f"Reason (keywords/lime summary): {mod_result.get('reason')}\n")

    # 2) Generate an appeal response for that same text
    print("---- Appeal response ----")
    appeal_text = generate_appeal_response(examples[0])
    print(appeal_text)

    # 3) Analyze false positives (print classifier behaviour on benign/ambiguous examples)
    print("---- False positives analysis ----")
    analyze_false_positives([
        "I hate when my code doesn't work",
        "This movie was terrible",
        "I strongly disagree with this policy"
    ])

    # 4) Batch moderate all examples and show a compact table-like output
    print("---- Batch moderation ----")
    batch_results = batch_moderate(examples)
    for r in batch_results:
        print(f"- '{r['input'][:60]}' -> remove={r['should_remove']}, class={r.get('class')}, severity={r.get('severity')}")
