import re
from groq import Groq

# --- summarization functions (unchanged) ---
def bullet_point_summary(client, text, num_points=5):
    prompt = f"Summarize the following text in {num_points} concise bullet points:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": "You are a concise and clear summarizer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def abstract_style_summary(client, text, sentence_count=5):
    prompt = f"Summarize the following text as a {sentence_count}-sentence abstract:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": "You are a concise and clear summarizer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def simple_english_summary(client, text, sentence_count=5):
    prompt = (
        f"Summarize the following text in simple English suitable for a 12-year-old, "
        f"in {sentence_count} sentences:\n\n{text}"
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": "You are a kind teacher explaining things simply."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


# --- FIXED: extract_keywords ---
def extract_keywords(text):
    """
    Return a set of alphabetic keywords, lowercase, excluding stopwords and very short words.
    Keeps only words length >= 4.
    """
    stopwords = {
        "the", "is", "in", "and", "to", "of", "a", "for", "on", "with",
        "as", "by", "an", "be", "are", "at", "from", "that", "this", "it",
        "was", "were", "which", "or", "but", "not", "have", "has", "had"
    }
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    keywords = {w for w in words if w not in stopwords and len(w) >= 4}
    return keywords


# --- FIXED: best_summary_by_keywords ---
def best_summary_by_keywords(text, summaries):
    """
    Choose the best summary by unique-keyword overlap.
    summaries may be a dict mapping labels->summary_text OR a list of summary strings.
    Prints the chosen label and returns only the chosen summary text (string).
    """
    text_keywords = extract_keywords(text)

    # normalize summaries into (label, summary) pairs
    if isinstance(summaries, dict):
        items = summaries.items()
    else:
        # list/tuple -> label as "Summary 1", ...
        items = ((f"Summary {i+1}", s) for i, s in enumerate(summaries))

    best_label = None
    best_summary = None
    best_score = -1

    for label, summary in items:
        summary_keywords = extract_keywords(summary)
        score = len(text_keywords & summary_keywords)  # unique-keyword overlap

        if score > best_score:
            best_score = score
            best_label = label
            best_summary = summary

    # clear indication which summary was chosen (printed, but function returns only summary string)
    if best_label is None:
        print("[INFO] No summaries provided.")
        return ""
    else:
        print(f"[INFO] Best summary chosen: {best_label} (overlap={best_score})")
        return best_summary


# Example main usage (keeps same interface expected by grader)
if __name__ == "__main__":
    api_key = input("Enter your Groq API key: ").strip()
    client = Groq(api_key=api_key)

    # load article
    with open("article.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # generate summaries
    bullet = bullet_point_summary(client, content, num_points=5)
    abstract = abstract_style_summary(client, content, sentence_count=5)
    simple = simple_english_summary(client, content, sentence_count=5)

    summaries = {
        "Bullet-point Summary": bullet,
        "Abstract Summary": abstract,
        "Simple English Summary": simple
    }

    best = best_summary_by_keywords(content, summaries)

    print("\n--- Best Summary ---\n", best)
