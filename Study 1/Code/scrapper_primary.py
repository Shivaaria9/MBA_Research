import re, os, time, random, requests, pandas as pd
from datetime import datetime, timezone, timedelta

TOTAL_TARGET = 1000
DATE_YEARS_BACK = 3
MIN_CHARS = 80
DELAY_MIN = 2.0
DELAY_MAX = 4.0
OUTPUT_DIR = "scarped_data"
OUTPUT_FILE ="primary.csv"

# Split target evenly across 4 subreddits
PER_SUB = TOTAL_TARGET // 4     # = 250 each

SUBREDDITS = {
    "Ozempic"    : {"per_sub": PER_SUB, "filter": False},  # dedicated forum — take all
    "semaglutide": {"per_sub": PER_SUB, "filter": False},  # dedicated forum — take all
    "weightloss" : {"per_sub": PER_SUB, "filter": True},   # general — keyword filter
    "diabetes"   : {"per_sub": PER_SUB, "filter": True},   # general — keyword filter
}

# Keywords used to filter r/weightloss and r/diabetes
KEYWORDS = [
    "ozempic", "semaglutide", "wegovy", "glp-1", "glp1",
    "tirzepatide", "mounjaro", "rybelsus",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]
 
API_URL = "https://api.pullpush.io/reddit/search/submission/"

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
 
def cutoff_ts():
    return int((datetime.now(timezone.utc)
                - timedelta(days=365 * DATE_YEARS_BACK)).timestamp())
 
def ts_to_date(ts):
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%d-%m-%Y")
    except Exception:
        return ""
 
def clean(text):
    if not text or str(text).strip() in ["[removed]", "[deleted]"]:
        return ""
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
 
def is_relevant(text, needs_filter):
    if not needs_filter:
        return True
    return any(kw in text.lower() for kw in KEYWORDS)
 
 
# ══════════════════════════════════════════════════════════════
#  API CALL  with retry
# ══════════════════════════════════════════════════════════════
 
def api_call(session, params, retries=4):
    for attempt in range(retries):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            resp = session.get(API_URL, params=params, timeout=30)
 
            if resp.status_code == 200:
                return resp.json().get("data", [])
            elif resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  ⏳ Rate limited — waiting {wait}s")
                time.sleep(wait)
            elif resp.status_code == 503:
                print(f"  ⚠  Server busy — waiting 30s")
                time.sleep(30)
            else:
                print(f"  ⚠  HTTP {resp.status_code} — retry {attempt+1}/{retries}")
                time.sleep(10 * (attempt + 1))
 
        except requests.exceptions.ConnectionError:
            print(f"  ⚠  Connection error — retry {attempt+1}/{retries}")
            time.sleep(15)
        except requests.exceptions.Timeout:
            print(f"  ⚠  Timeout — retry {attempt+1}/{retries}")
            time.sleep(10)
        except Exception as e:
            print(f"  ⚠  {e}")
            time.sleep(10)
 
    return []
# ══════════════════════════════════════════════════════════════
#  SCRAPE ONE SUBREDDIT
# ══════════════════════════════════════════════════════════════
def scrape_subreddit(session, subreddit, target, needs_filter):

    print(f"\n  ┌─ r/{subreddit}  (target: {target} posts)")

    cutoff = cutoff_ts()

    posts = []
    seen = set()

    # Dedicated subreddits
    if not needs_filter:

        search_terms = [None]

    # Selective/general subreddits
    else:

        search_terms = [
            "ozempic",
            "semaglutide",
            "wegovy",
            "mounjaro",
            "glp-1",
        ]

    for term in search_terms:

        before = int(datetime.now(timezone.utc).timestamp())

        empties = 0

        while len(posts) < target:

            params = {
                "subreddit": subreddit,
                "size": 100,
                "before": before,
                "after": cutoff,
                "sort": "desc",
                "sort_type": "created_utc",
                "is_self": "true",
            }

            # Add keyword query only for selective subs
            if term:
                params["q"] = term

            items = api_call(session, params)

            if not items:

                empties += 1

                if empties >= 3:
                    break

                time.sleep(8)
                continue

            empties = 0

            for item in items:

                if len(posts) >= target:
                    break

                pid = item.get("id", "")

                if pid in seen:
                    continue

                ts = item.get("created_utc", 0)

                title = clean(item.get("title", ""))

                body = clean(item.get("selftext", ""))

                full = f"{title} {body}"

                if len(full) < MIN_CHARS:
                    continue

                # Local keyword validation
                if needs_filter:

                    text_lower = full.lower()

                    if not any(
                        kw in text_lower
                        for kw in KEYWORDS
                    ):
                        continue

                seen.add(pid)

                posts.append({

                    "post_id": f"RED_{pid}",

                    "source_type": "Reddit",

                    "subreddit": f"r/{subreddit}",

                    "date": ts_to_date(ts),

                    "author": item.get(
                        "author",
                        "deleted"
                    ),

                    "title": title,

                    "text": body,

                    "combined_text": full,

                    "upvotes": item.get(
                        "score",
                        0
                    ),

                    "num_comments": item.get(
                        "num_comments",
                        0
                    ),

                    "flair": item.get(
                        "link_flair_text",
                        ""
                    ) or "",
                })

            oldest = min(
                i.get("created_utc", before)
                for i in items
            )

            if oldest >= before:
                break

            before = oldest - 1

            print(
                f"  │ {len(posts):>4} / {target} collected",
                end="\r"
            )

            time.sleep(random.uniform(2, 4))

    print(
        f"  └─ ✅ {len(posts)} posts collected "
        f"from r/{subreddit}"
    )

    return posts[:target] 

# ══════════════════════════════════════════════════════════════
#  CONNECTIVITY CHECK
# ══════════════════════════════════════════════════════════════
 
def check_connection():
    print("  Checking PullPush.io...")
    try:
        r = requests.get(
            API_URL,
            params={"subreddit": "Ozempic", "size": 1},
            timeout=15,
            headers={"User-Agent": USER_AGENTS[0]}
        )
        if r.status_code == 200 and r.json().get("data"):
            print("  ✅ PullPush.io is online\n")
            return True
        else:
            print(f"  ❌ PullPush returned HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cannot reach PullPush.io: {e}")
        return False

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
 
def run():
    print("=" * 62)
    print("  PRIMARY SCRAPER — Reddit via PullPush.io")
    print(f"  Target: {TOTAL_TARGET} posts ({PER_SUB} per subreddit)")
    print("=" * 62)
 
    if not check_connection():
        print("\n  PullPush.io is not reachable right now.")
        print("  → Try again in a few minutes (server may be temporarily down)")
        return None
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    session   = requests.Session()
    session.headers.update({"Accept": "application/json"})
 
    all_posts = []
 
    for subreddit, cfg in SUBREDDITS.items():
        posts = scrape_subreddit(
            session, subreddit, cfg["per_sub"], cfg["filter"]
        )
        all_posts.extend(posts)
 
    # ── Build final DataFrame ─────────────────────────────────
    df = (pd.DataFrame(all_posts)
            .drop_duplicates(subset=["post_id"])
            .reset_index(drop=True))
 
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
 
    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  ✅ DONE — PRIMARY DATA COLLECTED")
    print(f"{'='*62}")
    print(f"  Total rows     : {len(df)}")
    print(f"  Saved to       : {out_path}")
    print(f"\n  Subreddit breakdown:")
    for sub, n in df["subreddit"].value_counts().items():
        bar = "█" * int(n / df["subreddit"].value_counts().max() * 30)
        print(f"    {sub:<25} {bar} {n}")
    print(f"\n  Date range     : {df['date'].min()}  →  {df['date'].max()}")
    avg_len = df["combined_text"].str.len().mean()
    print(f"  Avg text length: {avg_len:.0f} characters")
    print(f"\n  → Next step: pass '{out_path}' to Analysis_1_AspectExtraction.py")
 
    return df
 
 
if __name__ == "__main__":
    run()