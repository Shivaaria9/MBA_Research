"""
╔══════════════════════════════════════════════════════════════════╗
║    SECONDARY SOURCE SCRAPER — Drugs.com + WebMD  [FIXED v3]     ║
║    Study : Cognitive Trade-Offs — Ozempic Discourse             ║
╠══════════════════════════════════════════════════════════════════╣
║  FIXED: Enhanced JavaScript waiting + dynamic content detection ║
║  The reviews are loaded via JavaScript → must wait longer        ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL:                                                        ║
║    pip install playwright beautifulsoup4 pandas lxml             ║
║    playwright install chromium                                  ║
║                                                                  ║
║  RUN:                                                            ║
║    python scraper_secondary_1000_rows_FIXED.py                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, re, time, random, json, pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ════════════════════════════════════════════════════════════════
#  ⚙️  SETTINGS
# ════════════════════════════════════════════════════════════════

DRUGSCOM_TARGET   = 500     # extract exactly 500 from Drugs.com
WEBMD_TARGET      = 500     # extract exactly 500 from WebMD

MIN_CHARS         = 80      # skip reviews shorter than this

HEADLESS          = False   # ← SET TO FALSE TO SEE THE BROWSER
                             # Watch what's loading → helps debug

DELAY_MIN         = 5.0     # seconds between page loads
DELAY_MAX         = 10.0    # seconds between page loads

EXTRA_PAUSE_EVERY = 5       # extra pause every N pages
EXTRA_PAUSE_SECS  = 25      # extra seconds added

CHECKPOINT_EVERY  = 100     # save progress every N reviews

OUTPUT_DIR        = "scraped_data"
OUTPUT_FILE       = "secondary.csv"

# ════════════════════════════════════════════════════════════════
#  DRUG PAGES
# ════════════════════════════════════════════════════════════════

DRUGSCOM_PAGES = [
    {
        "url"       : "https://www.drugs.com/comments/semaglutide/ozempic.html",
        "drug"      : "Ozempic",
        "condition" : "Type 2 Diabetes",
    },
    {
        "url"       : "https://www.drugs.com/comments/semaglutide/wegovy.html",
        "drug"      : "Wegovy",
        "condition" : "Weight Management",
    },
]

WEBMD_PAGES = [
    {
        "url"       : "https://www.webmd.com/drugs/drugreview-167386-"
                      "Ozempic+Subcutaneous.aspx",
        "drug_id"   : "167386",
        "drug_name" : "Ozempic+Subcutaneous",
        "drug"      : "Ozempic",
        "condition" : "Type 2 Diabetes / Weight Loss",
    },
    {
        "url"       : "https://www.webmd.com/drugs/drugreview-174491-"
                      "Wegovy+Subcutaneous.aspx",
        "drug_id"   : "174491",
        "drug_name" : "Wegovy+Subcutaneous",
        "drug"      : "Wegovy",
        "condition" : "Weight Management",
    },
]


# ════════════════════════════════════════════════════════════════
#  BROWSER SETUP
# ════════════════════════════════════════════════════════════════

VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
    {"width": 1280, "height": 800},
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) "
    "Gecko/20100101 Firefox/122.0",
]

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };
"""


def make_browser_context(playwright_instance):
    """Launch Chromium with stealth settings."""
    browser = playwright_instance.chromium.launch(
        headless=HEADLESS,
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-extensions",
        ],
    )
    context = browser.new_context(
        user_agent      = random.choice(USER_AGENTS),
        viewport        = random.choice(VIEWPORTS),
        locale          = "en-US",
        timezone_id     = "America/New_York",
        java_script_enabled = True,
        extra_http_headers  = {
            "Accept"         : "text/html,application/xhtml+xml,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    return browser, context


# ════════════════════════════════════════════════════════════════
#  PAGE FETCHER — ENHANCED FOR DYNAMIC CONTENT
# ════════════════════════════════════════════════════════════════

def fetch_page_drugscom(page, url, params=None, retries=3):
    """
    Fetch Drugs.com page with extensive JavaScript waiting.
    Reviews load dynamically — must wait for them to appear.
    """
    
    if params:
        qs  = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}" if "?" not in url else f"{url}&{qs}"

    for attempt in range(retries):
        try:
            print(f"      [Fetching {url.split('/')[-1][:40]}...]", flush=True)
            
            # Inject stealth
            page.add_init_script(STEALTH_JS)

            # Navigate with longer timeout
            page.goto(url, timeout=60000, wait_until="networkidle")
            
            # CRITICAL: Wait for the review container to appear
            # Drugs.com uses different selectors — try multiple
            try:
                page.wait_for_selector(".ddc-comment", timeout=15000)
                print(f"      [✓ Found .ddc-comment elements]", flush=True)
            except PWTimeout:
                print(f"      [⚠ .ddc-comment not found, trying alternatives...]", flush=True)
                try:
                    page.wait_for_selector("[data-comment-id]", timeout=10000)
                    print(f"      [✓ Found [data-comment-id] elements]", flush=True)
                except PWTimeout:
                    print(f"      [⚠ No review selectors found — page may be empty]", flush=True)

            # Scroll to trigger lazy-loading
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            time.sleep(2)
            
            # Scroll back to top
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            # Human-like pause
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            html = page.content()
            
            # Debug: Check if we got real content
            if len(html) < 5000:
                print(f"      [⚠ Very small HTML ({len(html)} bytes) — may be blocked]", flush=True)
            else:
                print(f"      [✓ Got {len(html)} bytes of HTML]", flush=True)
            
            return BeautifulSoup(html, "lxml")

        except PWTimeout as e:
            print(f"      [⚠ Timeout {attempt+1}/{retries}: {str(e)[:50]}]", flush=True)
            time.sleep(20)
        except Exception as e:
            print(f"      [⚠ Error {attempt+1}/{retries}: {str(e)[:50]}]", flush=True)
            time.sleep(15)

    return None


def fetch_page_webmd(page, url, params=None, retries=3):
    """
    Fetch WebMD page with extensive JavaScript waiting.
    WebMD uses different dynamic loading — must wait for userPost elements.
    """
    
    if params:
        qs  = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}" if "?" not in url else f"{url}&{qs}"

    for attempt in range(retries):
        try:
            print(f"      [Fetching WebMD page...]", flush=True)
            
            page.add_init_script(STEALTH_JS)
            page.goto(url, timeout=60000, wait_until="networkidle")
            
            # CRITICAL: Wait for review elements
            try:
                page.wait_for_selector(".userPost", timeout=15000)
                print(f"      [✓ Found .userPost elements]", flush=True)
            except PWTimeout:
                print(f"      [⚠ .userPost not found, trying alternatives...]", flush=True)
                try:
                    page.wait_for_selector("[data-postid]", timeout=10000)
                    print(f"      [✓ Found [data-postid] elements]", flush=True)
                except PWTimeout:
                    print(f"      [⚠ No review selectors found]", flush=True)

            # Scroll to load dynamic content
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            time.sleep(2)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

            html = page.content()
            
            if len(html) < 5000:
                print(f"      [⚠ Small HTML ({len(html)} bytes)]", flush=True)
            else:
                print(f"      [✓ Got {len(html)} bytes of HTML]", flush=True)
            
            return BeautifulSoup(html, "lxml")

        except PWTimeout as e:
            print(f"      [⚠ Timeout {attempt+1}/{retries}]", flush=True)
            time.sleep(20)
        except Exception as e:
            print(f"      [⚠ Error {attempt+1}/{retries}: {str(e)[:50]}]", flush=True)
            time.sleep(15)

    return None


def pause_extra(page_num=0):
    """Extra pause every N pages."""
    wait = random.uniform(DELAY_MIN, DELAY_MAX)
    if page_num > 0 and page_num % EXTRA_PAUSE_EVERY == 0:
        wait += EXTRA_PAUSE_SECS
        print(f"    [⏸ Extra {EXTRA_PAUSE_SECS}s pause at page {page_num}]", flush=True)
    time.sleep(wait)


# ════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════

def clean(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#\d+;", " ", text)
    return text.strip()

def parse_date_dc(raw):
    try:
        return datetime.strptime(raw.strip(), "%B %d, %Y").strftime("%d-%m-%Y")
    except Exception:
        return raw

def parse_date_wmd(raw):
    try:
        s = re.sub(r"[Ss]ubmitted:?\s*", "", raw).strip()
        return datetime.strptime(s, "%m/%d/%Y").strftime("%d-%m-%Y")
    except Exception:
        return raw

def save_checkpoint(records, label):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"_chk_{label}.csv")
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


# ════════════════════════════════════════════════════════════════
#  DRUGS.COM PARSER
# ════════════════════════════════════════════════════════════════

def parse_drugscom_page(soup, drug_info):
    reviews = []
    
    # Try multiple selectors
    blocks = soup.find_all("div", class_="ddc-comment")
    if not blocks:
        blocks = soup.find_all("div", attrs={"data-comment-id": True})
    if not blocks:
        blocks = soup.find_all("div", class_=re.compile(r"comment|review"))

    print(f"        Found {len(blocks)} review blocks", flush=True)

    for idx, block in enumerate(blocks):
        try:
            # Rating
            rating    = None
            rating_el = block.find(class_=re.compile(r"ddc-rating|rating"))
            if rating_el:
                m = re.search(r"width:\s*(\d+(?:\.\d+)?)\s*%",
                              rating_el.get("style", ""))
                if m:
                    rating = round(float(m.group(1)) / 10, 1)

            # Body
            body_el = block.find("p",
                class_=re.compile(r"comment.?text|ddc-comment-text"))
            if not body_el:
                paras = [p for p in block.find_all("p")
                         if len(p.get_text(strip=True)) > 50]
                body_el = paras[0] if paras else None
            
            body = clean(body_el.get_text()) if body_el else ""
            if len(body) < MIN_CHARS:
                continue

            # Title
            t_el  = block.find(class_=re.compile(r"title|headline"))
            title = clean(t_el.get_text()) if t_el else ""

            # Date
            d_el = block.find(class_=re.compile(r"date|posted"))
            date_str = parse_date_dc(clean(d_el.get_text())) if d_el else ""

            # Condition
            c_el = block.find(class_=re.compile(r"condition"))
            cond = clean(c_el.get_text()) if c_el else drug_info["condition"]
            cond = re.sub(r"^[Cc]ondition:?\s*", "", cond).strip() or drug_info["condition"]

            # User
            u_el = block.find(class_=re.compile(r"user"))
            user = re.sub(r"\s+", "_", clean(u_el.get_text()).lower())[:30] if u_el else f"user_{idx+1}"

            reviews.append({
                "title"    : title,
                "text"     : body,
                "rating"   : rating,
                "date"     : date_str,
                "user_id"  : user,
                "condition": cond,
                "drug"     : drug_info["drug"],
            })
        except Exception as e:
            continue

    return reviews


def scrape_drugscom(page):
    print(f"\n  ┌─ Drugs.com  (target: {DRUGSCOM_TARGET})")
    all_reviews = []

    for drug_info in DRUGSCOM_PAGES:
        if len(all_reviews) >= DRUGSCOM_TARGET:
            break

        print(f"  │")
        print(f"  ├─ Drug: {drug_info['drug']}")
        pg = 1

        while len(all_reviews) < DRUGSCOM_TARGET:
            print(f"  │    Page {pg:3d}", end="  ", flush=True)

            url    = drug_info["url"]
            params = {"page": pg} if pg > 1 else None

            soup = fetch_page_drugscom(page, url, params)

            if soup is None:
                print("→ Failed", flush=True)
                break

            batch = parse_drugscom_page(soup, drug_info)
            if not batch:
                print("→ No reviews found", flush=True)
                break

            all_reviews.extend(batch)
            print(f"→ +{len(batch):2d}  │  total: {len(all_reviews)}", flush=True)

            if len(all_reviews) % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_reviews, "drugscom")

            if len(all_reviews) >= DRUGSCOM_TARGET:
                break

            pg += 1
            pause_extra(pg)

    all_reviews = all_reviews[:DRUGSCOM_TARGET]
    print(f"  └─ ✅  Drugs.com: {len(all_reviews)} reviews")
    return all_reviews


# ════════════════════════════════════════════════════════════════
#  WEBMD PARSER
# ════════════════════════════════════════════════════════════════

def parse_webmd_page(soup, drug_info):
    reviews = []
    
    blocks = soup.find_all("div", class_="userPost")
    if not blocks:
        blocks = soup.find_all("div", attrs={"data-postid": True})
    if not blocks:
        blocks = soup.find_all("div", class_=re.compile(r"post|review"))

    print(f"        Found {len(blocks)} review blocks", flush=True)

    for idx, block in enumerate(blocks):
        try:
            # Rating
            rating = None
            r_el = block.find(class_=re.compile(r"rating"))
            if not r_el:
                r_el = block.find("span", style=re.compile(r"width:\s*\d+%"))
            if r_el:
                m = re.search(r"width:\s*(\d+(?:\.\d+)?)\s*%",
                              r_el.get("style", ""))
                if m:
                    rating = round(float(m.group(1)) / 10, 1)

            # Body
            body_el = block.find(id=re.compile(r"comFull|reviewFull"))
            if not body_el:
                body_el = block.find(class_=re.compile(r"description|text"))
            if not body_el:
                paras = block.find_all("p")
                body_el = max(paras, key=lambda p: len(p.get_text()), default=None)
            
            body = clean(body_el.get_text()) if body_el else ""
            if len(body) < MIN_CHARS:
                continue

            # Title
            t_el = block.find(class_=re.compile(r"title|headline"))
            title = clean(t_el.get_text()) if t_el else ""

            # Date
            d_el = block.find(class_=re.compile(r"date"))
            date_str = parse_date_wmd(clean(d_el.get_text())) if d_el else ""

            # Condition
            c_el = block.find(class_=re.compile(r"condition"))
            cond = clean(c_el.get_text()) if c_el else drug_info["condition"]
            cond = re.sub(r"^[Cc]ondition:?\s*", "", cond).strip() or drug_info["condition"]

            # User
            u_el = block.find(class_=re.compile(r"user|author"))
            user = re.sub(r"\s+", "_", clean(u_el.get_text()).lower())[:30] if u_el else f"reviewer_{idx+1}"

            reviews.append({
                "title"    : title,
                "text"     : body,
                "rating"   : rating,
                "date"     : date_str,
                "user_id"  : user,
                "condition": cond,
                "drug"     : drug_info["drug"],
            })
        except Exception:
            continue

    return reviews


def scrape_webmd(page):
    print(f"\n  ┌─ WebMD  (target: {WEBMD_TARGET})")
    all_reviews = []

    for drug_info in WEBMD_PAGES:
        if len(all_reviews) >= WEBMD_TARGET:
            break

        print(f"  │")
        print(f"  ├─ Drug: {drug_info['drug']}")

        page_idx = 0
        empty_streak = 0

        while len(all_reviews) < WEBMD_TARGET:
            print(f"  │    Page {page_idx:3d}", end="  ", flush=True)

            base = drug_info["url"]
            params = {
                "drugid"         : drug_info["drug_id"],
                "drugname"       : drug_info["drug_name"],
                "pageIndex"      : page_idx,
                "sortby"         : "3",
                "conditionFilter": "-500",
            }

            soup = fetch_page_webmd(page, base, params)

            if soup is None:
                print("→ Failed", flush=True)
                break

            batch = parse_webmd_page(soup, drug_info)

            if not batch:
                empty_streak += 1
                print(f"→ Empty ({empty_streak}/3)", flush=True)
                if empty_streak >= 3:
                    break
                time.sleep(20)
                page_idx += 1
                continue

            empty_streak = 0
            all_reviews.extend(batch)
            print(f"→ +{len(batch):2d}  │  total: {len(all_reviews)}", flush=True)

            if len(all_reviews) % CHECKPOINT_EVERY == 0:
                save_checkpoint(all_reviews, "webmd")

            if len(all_reviews) >= WEBMD_TARGET:
                break

            page_idx += 1
            pause_extra(page_idx)

    all_reviews = all_reviews[:WEBMD_TARGET]
    print(f"  └─ ✅  WebMD: {len(all_reviews)} reviews")
    return all_reviews


# ════════════════════════════════════════════════════════════════
#  BUILD DATAFRAME
# ════════════════════════════════════════════════════════════════

def build_dataframe(dc_reviews, wmd_reviews):
    records = []
    counter = 1
    for platform, source in [("Drugs.com", dc_reviews),
                               ("WebMD",     wmd_reviews)]:
        for r in source:
            records.append({
                "review_id"     : f"REV_{counter:06d}",
                "source_type"   : "Review_Platform",
                "platform"      : platform,
                "drug"          : r.get("drug", ""),
                "condition"     : r.get("condition", ""),
                "date"          : r.get("date", ""),
                "user_id"       : r.get("user_id", ""),
                "title"         : r.get("title", ""),
                "text"          : r.get("text", ""),
                "combined_text" : (r.get("title","") + " " +
                                   r.get("text","")).strip(),
                "rating"        : r.get("rating"),
            })
            counter += 1

    df = pd.DataFrame(records)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").round(1)
    return df


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def run():
    total_target = DRUGSCOM_TARGET + WEBMD_TARGET

    print("=" * 66)
    print("  SECONDARY SCRAPER — Drugs.com + WebMD  [FIXED v3]")
    print(f"  Target : {total_target:,} reviews total")
    print(f"           {DRUGSCOM_TARGET} Drugs.com  +  {WEBMD_TARGET} WebMD")
    print(f"  Mode   : {'VISIBLE browser (debugging)' if not HEADLESS else 'headless'}")
    print(f"  Delay  : {DELAY_MIN}–{DELAY_MAX}s between pages")
    print("=" * 66)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with sync_playwright() as pw:
        browser, context = make_browser_context(pw)
        page = context.new_page()

        # ── Drugs.com ────────────────────────────────────────
        dc_reviews = scrape_drugscom(page)
        if dc_reviews:
            pd.DataFrame(dc_reviews).to_csv(
                os.path.join(OUTPUT_DIR, "drugscom_raw.csv"),
                index=False, encoding="utf-8-sig")

        # Pause + fresh page
        print("\n  Pausing 60s before WebMD...")
        time.sleep(60)
        page.close()
        page = context.new_page()

        # ── WebMD ─────────────────────────────────────────────
        wmd_reviews = scrape_webmd(page)
        if wmd_reviews:
            pd.DataFrame(wmd_reviews).to_csv(
                os.path.join(OUTPUT_DIR, "webmd_raw.csv"),
                index=False, encoding="utf-8-sig")

        browser.close()

    # ── Combine ───────────────────────────────────────────────
    if not dc_reviews and not wmd_reviews:
        print("\n  ✗ No reviews collected.")
        print("  DEBUG CHECKLIST:")
        print("    1. HEADLESS=False is set (you should see the browser)")
        print("    2. Watch the browser window — does page load?")
        print("    3. Look for review elements on the page")
        print("    4. Check browser console for JavaScript errors")
        print("    5. URL may have changed → try opening in your browser")
        return None

    df = build_dataframe(dc_reviews or [], wmd_reviews or [])
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    total = len(df)
    print(f"\n{'='*66}")
    print(f"  ✅ DONE — SECONDARY DATA COLLECTED")
    print(f"  {'='*66}")
    print(f"  Total reviews  : {total}")
    print(f"  Saved to       : {out_path}")
    
    if total > 0:
        print(f"\n  Platform split:")
        for p, n in df["platform"].value_counts().items():
            bar = "█" * int(n / total * 30)
            pct = (n / total * 100)
            print(f"    {p:<12} {bar} {n}  ({pct:.1f}%)")
        print(f"\n  Drug split:")
        for d, n in df["drug"].value_counts().items():
            print(f"    {d:<30} {n}")
        if df["rating"].notna().any():
            print(f"\n  Avg rating : {df['rating'].mean():.1f} / 10")
        avg_len = df["combined_text"].str.len().mean()
        print(f"  Avg text length : {avg_len:.0f} chars")

    for label in ["drugscom", "webmd"]:
        chk = os.path.join(OUTPUT_DIR, f"_chk_{label}.csv")
        if os.path.exists(chk):
            os.remove(chk)

    print(f"\n  → Next: run Analysis_1_AspectExtraction.py")
    return df


if __name__ == "__main__":
    run()