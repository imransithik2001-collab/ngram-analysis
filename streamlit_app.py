import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
import re
from collections import Counter

# =======================
# NLTK Setup
# =======================
# Ensure required NLTK resources are available
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

STOPWORDS = set(stopwords.words("english"))

# =======================
# Scraping Functions
# =======================
def scrape_webpage(url):
    """Scrape both meta title and on-page <p> content from a URL."""
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Meta title
    title_tag = soup.find("title")
    meta_title = title_tag.get_text().strip() if title_tag else ""

    # On-page content
    paragraphs = soup.find_all("p")
    page_text = " ".join([p.get_text() for p in paragraphs])

    return {"url": url, "meta_title": meta_title, "page_text": page_text}


def clean_text(text):
    """Lowercase + remove punctuation/numbers."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters/spaces
    text = text.lower()
    return text


def tokenize(text):
    """Tokenize and remove stopwords."""
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS]


def run_ngram_analysis(text, n=2, top_k=10):
    """Generate n-grams and return most common ones."""
    tokens = tokenize(clean_text(text))
    ngram_list = list(ngrams(tokens, n))
    return Counter(ngram_list).most_common(top_k)


# =======================
# Streamlit App
# =======================
st.set_page_config(page_title="N-gram Analyzer", layout="wide")
st.title("🔍 N-gram Analyzer for Meta Title & On-page Content")

st.sidebar.header("Configuration")
analysis_type = st.sidebar.radio(
    "Select what to analyze:", ["Meta Title", "On-page Content"]
)
ngram_size = st.sidebar.selectbox("N-gram size:", [1, 2, 3], index=1)
top_k = st.sidebar.slider("Number of top n-grams:", 5, 20, 10)
combine_pages = st.sidebar.checkbox("Combine all pages", value=True)

# Input URLs

urls_input = st.text_area(
    "Enter URLs (one per line):",
    placeholder="https://example.com",
    height=150,
)

urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

if st.button("Scrape & Analyze"):
    if not urls:
        st.error("Please enter at least one URL.")
    else:
        # Scrape all URLs once and store results
        scraped_data = [scrape_webpage(u) for u in urls]

        # Save results in session_state to avoid re-scraping
        st.session_state["scraped_data"] = scraped_data
        st.success("✅ Scraping complete. Now you can switch filters.")

# Load stored data if available
if "scraped_data" in st.session_state:
    scraped_data = st.session_state["scraped_data"]

    # Helper function: select text based on filter
    def select_text(entry):
        return entry["meta_title"] if analysis_type == "Meta Title" else entry["page_text"]

    if combine_pages:
        combined_text = " ".join([select_text(e) for e in scraped_data])
        freqs = run_ngram_analysis(combined_text, n=ngram_size, top_k=top_k)
        df = pd.DataFrame([(" ".join(ng), freq) for ng, freq in freqs], columns=["N-gram", "Frequency"])
        st.subheader(f"Top {top_k} {ngram_size}-grams ({analysis_type}, Combined Pages)")
        st.dataframe(df)
        st.bar_chart(df.set_index("N-gram"))
    else:
        for entry in scraped_data:
            text = select_text(entry)
            freqs = run_ngram_analysis(text, n=ngram_size, top_k=top_k)
            df = pd.DataFrame([(" ".join(ng), freq) for ng, freq in freqs], columns=["N-gram", "Frequency"])
            st.subheader(f"Top {top_k} {ngram_size}-grams ({analysis_type}) - {entry['url']}")
            st.dataframe(df)
            st.bar_chart(df.set_index("N-gram"))
