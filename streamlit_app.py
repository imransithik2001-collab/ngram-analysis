import time
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt

# === Stopwords handling (fallback if nltk unavailable) ===
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set([
        "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
        "that", "this", "as", "at", "by", "an", "be", "are", "or", "from"
    ])


# === Scraping functions ===
def scrape_url(url, include_headings=False):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return {"url": url, "error": str(e), "meta_title": "", "content": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Meta title
    meta_title = soup.title.string.strip() if soup.title else ""

    # On-page content
    texts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    if include_headings:
        texts += [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]

    content = " ".join(texts)
    return {"url": url, "meta_title": meta_title, "content": content}


# === N-gram analysis ===
def tokenize(text):
    return [
        w.lower() for w in text.split()
        if w.isalpha() or w.replace("-", "").isalpha()
    ]


def get_ngrams(text, n=2, top_k=20):
    words = [w for w in tokenize(text) if w not in STOPWORDS]
    ngrams = zip(*[words[i:] for i in range(n)])
    ngram_list = [" ".join(ng) for ng in ngrams]
    return Counter(ngram_list).most_common(top_k)


# === Streamlit App ===
st.set_page_config(page_title="N-gram Analyzer", page_icon="🔍", layout="wide")
st.title("🔍 N-gram Analyzer for Meta Titles & On-page Content")

st.sidebar.header("Settings")
urls_input = st.sidebar.text_area("Enter URLs (one per line)", height=150)
analyze_mode = st.sidebar.selectbox("Analyze", ["Meta title", "On-page content", "Both"])
include_headings = st.sidebar.checkbox("Include H1/H2/H3 in on-page content", True)
n = st.sidebar.slider("N-gram size (n)", 1, 4, 2)
top_k = st.sidebar.slider("Top-K results", 5, 50, 20)
combine_mode = st.sidebar.checkbox("Combine all pages", True)
delay = st.sidebar.slider("Delay between requests (seconds)", 0, 5, 1)

run_btn = st.sidebar.button("Run analysis")

if run_btn:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        results = []
        meta_titles = []
        combined_content = []

        st.info("Scraping started...")
        progress = st.progress(0)
        for i, url in enumerate(urls):
            res = scrape_url(url, include_headings=include_headings)
            if res.get("error"):
                st.error(f"Error scraping {url}: {res['error']}")
                continue

            if analyze_mode in ["On-page content", "Both"]:
                combined_content.append(res["content"])
            if analyze_mode in ["Meta title", "Both"]:
                meta_titles.append({"URL": url, "Meta Title": res["meta_title"]})

            results.append(res)
            progress.progress((i + 1) / len(urls))
            time.sleep(delay)

        # Meta titles
        if analyze_mode in ["Meta title", "Both"] and meta_titles:
            st.subheader("Meta Titles")
            st.dataframe(pd.DataFrame(meta_titles))

        # N-gram analysis
        if analyze_mode in ["On-page content", "Both"]:
            if combine_mode:
                st.subheader("Combined N-gram Analysis")
                all_text = " ".join(combined_content)
                ngrams = get_ngrams(all_text, n=n, top_k=top_k)
                df = pd.DataFrame(ngrams, columns=["N-gram", "Frequency"])
                st.dataframe(df)

                # Chart
                fig, ax = plt.subplots()
                df.plot(kind="barh", x="N-gram", y="Frequency", ax=ax, legend=False)
                st.pyplot(fig)

                # Download
                st.download_button(
                    "Download CSV", df.to_csv(index=False), "ngrams.csv", "text/csv"
                )
            else:
                st.subheader("Per-page N-gram Analysis")
                for res in results:
                    text = res["content"]
                    if not text.strip():
                        continue
                    ngrams = get_ngrams(text, n=n, top_k=top_k)
                    df = pd.DataFrame(ngrams, columns=["N-gram", "Frequency"])
                    st.markdown(f"**{res['url']}**")
                    st.dataframe(df)

                    fig, ax = plt.subplots()
                    df.plot(kind="barh", x="N-gram", y="Frequency", ax=ax, legend=False)
                    st.pyplot(fig)

                    st.download_button(
                        f"Download CSV for {res['url']}",
                        df.to_csv(index=False),
                        f"ngrams_{res['url'].replace('https://','').replace('/','_')}.csv",
                        "text/csv",
                    )
