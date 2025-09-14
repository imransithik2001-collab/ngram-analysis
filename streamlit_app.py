import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk

# Ensure NLTK downloads work locally
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.util import ngrams

STOPWORDS = set(stopwords.words("english"))

# ==============================
# Scraper (cached)
# ==============================
@st.cache_data
def scrape_url(url: str, include_headings=False):
    """Scrape meta title and on-page content from a URL (cached)."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Meta title
        meta_title = None
        if soup.title:
            meta_title = soup.title.string.strip()
        elif soup.find("meta", attrs={"name": "title"}):
            meta_title = soup.find("meta", attrs={"name": "title"})["content"].strip()

        # On-page content
        content = " ".join([p.get_text() for p in soup.find_all("p")])
        if include_headings:
            headings = " ".join(
                [h.get_text() for h in soup.find_all(["h1", "h2", "h3"])]
            )
            content = content + " " + headings

        return {"url": url, "meta_title": meta_title, "content": content}
    except Exception:
        return {"url": url, "meta_title": None, "content": ""}


# ==============================
# Text processing
# ==============================
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS]


def run_ngram_analysis(text, n=2, top_k=10):
    tokens = tokenize(clean_text(text))
    ngram_list = list(ngrams(tokens, n))
    ngram_freq = Counter(ngram_list).most_common(top_k)
    return ngram_freq


# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="N-gram Analyzer", layout="wide")
st.title("🔎 N-gram Analyzer (Meta Title & On-page Content)")

# Sidebar
st.sidebar.header("Settings")
urls_input = st.sidebar.text_area(
    "Enter URLs (one per line)", height=200, placeholder="https://example.com"
)

analysis_mode = st.sidebar.radio(
    "Analyze", ["Meta title", "On-page", "Both"], index=1
)

include_headings = st.sidebar.checkbox("Include H1/H2/H3 in On-page", value=False)
combine_pages = st.sidebar.checkbox("Combine all pages into one analysis", value=True)
ngram_size = st.sidebar.slider("N-gram size", 1, 4, 2)
top_k = st.sidebar.slider("Top-K results", 5, 30, 10)

if st.sidebar.button("Run Analysis"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if not urls:
        st.error("⚠️ Please enter at least one URL")
        st.stop()

    # Scrape all URLs (cached)
    scraped_data = [scrape_url(u, include_headings) for u in urls]

    # Show raw meta titles (for reference)
    if analysis_mode in ["Meta title", "Both"]:
        st.subheader("📌 Meta Titles Extracted")
        df_titles = pd.DataFrame(
            [(d["url"], d["meta_title"]) for d in scraped_data],
            columns=["URL", "Meta Title"],
        )
        st.dataframe(df_titles, use_container_width=True)

    # Select text to analyze
    def select_text(entry):
        if analysis_mode == "Meta title":
            return entry["meta_title"] or ""
        elif analysis_mode == "On-page":
            return entry["content"] or ""
        else:  # Both
            return " ".join(
                [entry["meta_title"] or "", entry["content"] or ""]
            )

    if combine_pages:
        combined_text = " ".join([select_text(e) for e in scraped_data])
        freqs = run_ngram_analysis(combined_text, n=ngram_size, top_k=top_k)
        df = pd.DataFrame(
            [(" ".join(ng), freq) for ng, freq in freqs],
            columns=["N-gram", "Frequency"],
        )

        st.subheader("📊 Combined Results")
        st.dataframe(df, use_container_width=True)

        # Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df["N-gram"], df["Frequency"], color="skyblue")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Download
        st.download_button(
            "Download CSV", df.to_csv(index=False).encode("utf-8"), "ngrams.csv", "text/csv"
        )
    else:
        st.subheader("📊 Per-page Results")
        for entry in scraped_data:
            text = select_text(entry)
            if not text.strip():
                st.warning(f"No content found for {entry['url']}")
                continue

            freqs = run_ngram_analysis(text, n=ngram_size, top_k=top_k)
            df = pd.DataFrame(
                [(" ".join(ng), freq) for ng, freq in freqs],
                columns=["N-gram", "Frequency"],
            )

            st.markdown(f"**URL:** {entry['url']}")
            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(df["N-gram"], df["Frequency"], color="lightgreen")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.download_button(
                f"Download CSV for {entry['url']}",
                df.to_csv(index=False).encode("utf-8"),
                f"ngrams_{entry['url'].replace('https://','').replace('/','_')}.csv",
                "text/csv",
            )
