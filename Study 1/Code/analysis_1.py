'''this code is for both big and small file '''
import os, re, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

PRIMARY_FILE   = "original Data prelims/primary.csv"      # ← full scale
SECONDARY_FILE = "original Data prelims/secondary.csv"  # ← full scale
OUTPUT_DIR     = "A_1_results"

# ── FastText settings ─────────────────────────────────────
# These are tuned differently for pilot vs full scale.
# Larger corpus → lower similarity threshold (catches more synonyms)
#               → higher top_n (checks more neighbours per seed word)
#               → lower min_count (rarer slang words still get vectors)
#
#   PILOT  (1,000–2,000 rows):
#     FASTTEXT_MIN_CORPUS = 500   FASTTEXT_SIMILARITY = 0.65
#     FASTTEXT_TOP_N = 20         FASTTEXT_MIN_COUNT  = 3
#
#   FULL SCALE  (20,000+ rows):  ← current values below
#     FASTTEXT_MIN_CORPUS = 5000  FASTTEXT_SIMILARITY = 0.60
#     FASTTEXT_TOP_N = 30         FASTTEXT_MIN_COUNT  = 2

FASTTEXT_MIN_CORPUS   = 5000   # activate FastText only if corpus ≥ this
FASTTEXT_SIMILARITY   = 0.60   # lower than pilot → discovers more synonyms
                                # safe at full scale because embeddings are richer
FASTTEXT_TOP_N        = 30     # check more neighbours → catches more Reddit slang
FASTTEXT_MIN_COUNT    = 2      # rare slang (ozempic face, food noise) still vectorised

# Fuzzy matching threshold — same for both scales
FUZZY_THRESHOLD       = 85

TEXT_COLUMN    = "combined_text"   # column containing the text to analyse
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════╗
# ║   ASPECT DICTIONARY — SEED TERMS                            ║
# ║                                                              ║
# ║   Each key  = one aspect category                           ║
# ║   Each list = seed words (clinical + informal + slang)      ║
# ║   FastText will EXPAND these from your actual corpus.       ║
# ╚══════════════════════════════════════════════════════════════╝

BENEFITS = {

    "weight_loss": [
        # Clinical / formal
        "weight loss", "weight reduction", "weight management", "bmi",
        "obesity", "overweight", "body weight", "body mass",
        # Measurements
        "pounds", "lbs", "kilos", "kg", "stone",
        # Verbs / informal
        "lost weight", "lose weight", "losing weight",
        "dropped", "shed", "slim", "slimmer", "lighter",
        "down lbs", "down pounds", "down kg",
        # Reddit slang
        "skinny", "thin", "smaller", "fit into", "clothes fit",
        "inches off", "waist smaller", "size down",
    ],

    "appetite_suppression": [
        # Clinical
        "appetite", "satiety", "satiation", "anorexigenic",
        # Common
        "hunger", "cravings", "craving", "snacking", "overeating",
        "portion", "full", "fullness", "eat less", "eating less",
        # Reddit community terms
        "food noise",          # ← very common on r/Ozempic — dictionary only would miss this
        "food cravings gone",
        "not hungry",
        "forgot to eat",
        "no appetite",
        "stopped wanting food",
        "quiet the hunger",
        "food thoughts",
    ],

    "glucose_control": [
        # Clinical
        "glucose", "blood glucose", "blood sugar", "glycemic",
        "a1c", "hba1c", "hemoglobin a1c", "fasting glucose",
        "insulin", "insulin resistance", "hyperglycemia",
        "glycemic control", "type 2", "type2", "t2d", "t2dm",
        # Common
        "diabetes", "diabetic", "sugar levels", "sugar control",
        "sugar reading", "sugar spike",
        # Devices
        "cgm", "glucometer", "continuous glucose", "glucose monitor",
    ],

    "energy_wellbeing": [
        # Energy
        "energy", "energetic", "vitality", "active", "moving more",
        "exercise more", "working out more",
        # Mental / emotional
        "feel better", "feeling better", "mood", "mental clarity",
        "brain fog lifted", "clearer head",
        # Identity
        "confidence", "confident", "self-esteem", "self-worth",
        "quality of life", "better life", "life changed",
        # Physical
        "inflammation", "joint pain reduced", "mobility", "move easier",
    ],

    "cardiovascular": [
        "heart", "cardiovascular", "cardiac",
        "blood pressure", "hypertension", "bp lowered",
        "cholesterol", "ldl", "hdl", "triglycerides",
        "heart health", "heart disease", "heart risk",
        "stroke risk", "heart attack risk",
        "reduced cardiovascular", "cardioprotective",
    ],
}


SIDE_EFFECTS = {

    "nausea_vomiting": [
        # Clinical
        "nausea", "vomiting", "emesis", "antiemetic",
        # Common
        "nauseous", "nauseated", "sick to my stomach", "stomach sick",
        "vomit", "threw up", "throw up", "thrown up",
        # Reddit slang
        "queasy", "barf", "barfing", "barfed",    # ← missed by clinical dictionary
        "puking", "puke", "puked",
        "retching", "heaving", "dry heaving",
        "sick feeling", "feel sick", "feeling sick",
        "can't keep food down", "nothing stays down",
        "morning sickness",
    ],

    "gastrointestinal": [
        # Clinical
        "diarrhea", "diarrhoea", "constipation", "gastroparesis",
        "flatulence", "abdominal pain", "gastric",
        # Common
        "constipated", "loose stool", "watery stool", "runny stool",
        "bloating", "bloated", "gas", "gassy",
        "stomach pain", "stomach cramps", "stomach ache", "stomach issues",
        "bowel", "bowel issues", "gi issues", "digestive issues",
        "heartburn", "acid reflux", "indigestion", "reflux",
        # Reddit community terms
        "sulfur burps",        # ← extremely common on Reddit — clinical dict misses this
        "egg burps",
        "burping", "belching",
        "gut issues", "gut problems",
    ],

    "fatigue": [
        # Clinical
        "fatigue", "lethargy", "asthenia", "malaise",
        # Common
        "tired", "tiredness", "exhausted", "exhaustion",
        "weak", "weakness", "sluggish", "drained",
        "no energy", "low energy", "wiped out",
        # Reddit informal
        "tired all the time",
        "can't function",
        "feel like a zombie",  # ← vivid Reddit expression
        "dead tired",
        "bed ridden", "bed rest",
    ],

    "injection_site": [
        # Clinical
        "injection site", "subcutaneous reaction", "lipodystrophy",
        # Common
        "injection pain", "needle pain", "bruising", "bruise",
        "redness", "swelling", "lump", "bump", "knot",
        "site reaction", "site pain", "painful injection",
        "hurts to inject", "injection hurts",
        "auto-injector", "injector pen", "pen hurts",
    ],

    "hair_muscle_face": [
        # Clinical
        "alopecia", "sarcopenia", "muscle wasting", "muscle atrophy",
        "lean mass loss",
        # Common
        "hair loss", "hair thinning", "losing hair", "hair falling out",
        "muscle loss", "losing muscle",
        # Reddit community terms
        "ozempic face",        # ← coined by Reddit community
        "ozempic butt",
        "face sagging", "face sunken", "gaunt face",
        "hollow cheeks", "facial fat loss",
        "face aging", "looking older",
        "skin sagging", "loose skin",
    ],

    "psychological": [
        # Clinical
        "anxiety disorder", "depression", "mood disorder",
        "suicidal ideation", "mental health",
        # Common
        "anxiety", "anxious", "depressed", "depression",
        "mood swings", "irritable", "irritability",
        "emotional", "crying", "cry a lot",
        # Cognitive
        "brain fog",           # ← informal but very common
        "memory issues", "forgetful", "foggy thinking",
        "can't concentrate", "concentration issues",
        # Food relationship
        "disordered eating", "food relationship", "obsessed with food",
    ],

    "serious_events": [
        # Pancreatitis
        "pancreatitis", "inflamed pancreas", "pancreas pain",
        "severe stomach pain", "severe back pain",
        # Thyroid
        "thyroid cancer", "thyroid tumour", "medullary thyroid",
        "thyroid warning",
        # Other serious
        "kidney failure", "kidney disease", "gallbladder",
        "gallstones", "hospitalised", "hospitalized", "er visit",
        "emergency room", "serious side effect",
    ],
}

# All aspects combined — used in matching loop
ALL_ASPECTS = {**BENEFITS, **SIDE_EFFECTS}


# ╔══════════════════════════════════════════════════════════════╗
# ║   LAYER 2 — FASTTEXT SEMANTIC EXPANSION                     ║
# ╚══════════════════════════════════════════════════════════════╝

def train_fasttext(texts):
    """
    Train FastText on YOUR corpus.
    Why train on your data instead of using a pre-trained model?
    → Pre-trained models (Google News) don't know "food noise",
      "ozempic face", "sulfur burps". Your corpus does.
    Returns model or None if gensim not available.
    """
    try:
        from gensim.models import FastText
        print("  [Layer 2] Training FastText on corpus...")

        sentences = []
        for text in texts:
            if isinstance(text, str):
                tokens = re.findall(r"[a-z]{2,}", text.lower())
                if tokens:
                    sentences.append(tokens)

        if len(sentences) < 200:
            print(f"  [Layer 2] Too few sentences ({len(sentences)}) — skipping FastText")
            return None

        model = FastText(
            sentences   = sentences,
            vector_size = 150,    # larger at full scale → richer vectors
            window      = 6,      # wider context window → better Reddit slang capture
            min_count   = FASTTEXT_MIN_COUNT,
            workers     = 4,
            epochs      = 15,     # more epochs at full scale → better embeddings
            sg          = 1,      # skip-gram: better for rare/informal words
            min_n       = 3,      # character n-gram min size
            max_n       = 6,      # character n-gram max size → handles misspellings
        )
        print(f"  [Layer 2] ✅ FastText trained — vocabulary: {len(model.wv):,} words")
        return model

    except ImportError:
        print("  [Layer 2] gensim not installed — skipping FastText expansion")
        print("            Install: pip install gensim")
        return None
    except Exception as e:
        print(f"  [Layer 2] FastText failed: {e} — using seed dictionary only")
        return None


def expand_with_fasttext(model, seed_dict, threshold, top_n):
    """
    For each seed word, find semantically similar words in the corpus.
    Add them to the keyword list if cosine similarity ≥ threshold.
    Returns expanded dictionary.
    """
    expanded = {}

    for aspect, seeds in seed_dict.items():
        new_terms = set()

        for seed in seeds:
            # For multi-word seeds, use the last meaningful word as query
            words = re.findall(r"[a-z]{3,}", seed.lower())
            if not words:
                continue
            query = words[-1]

            try:
                similar = model.wv.most_similar(query, topn=top_n)
                for word, score in similar:
                    if (score >= threshold
                            and len(word) >= 3
                            and re.match(r"^[a-z]+$", word)):
                        new_terms.add(word)
            except KeyError:
                pass   # word not in vocabulary — that's fine

        original   = set(seeds)
        discovered = new_terms - original
        expanded[aspect] = list(original | new_terms)

        if discovered:
            sample = sorted(discovered)[:6]
            print(f"    [{aspect}] discovered: {', '.join(sample)}"
                  f"{'...' if len(discovered) > 6 else ''}"
                  f"  (+{len(discovered)} terms)")

    return expanded


# ╔══════════════════════════════════════════════════════════════╗
# ║   LAYER 3 — FUZZY MATCHING                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def make_fuzzy_checker(keyword_list):
    """
    Returns a function that checks if any token in a text
    fuzzy-matches any keyword in the list.
    Catches: nausia→nausea, diareah→diarrhea, constipashun→constipation
    """
    try:
        from rapidfuzz import process, fuzz

        # Only single words are relevant for fuzzy matching
        single_words = [kw for kw in keyword_list
                        if len(kw.split()) == 1 and len(kw) >= 4]

        def checker(tokens):
            for token in tokens:
                if len(token) < 4:
                    continue
                match = process.extractOne(
                    token, single_words,
                    scorer    = fuzz.ratio,
                    score_cutoff = FUZZY_THRESHOLD,
                )
                if match:
                    return True
            return False

        return checker

    except ImportError:
        # rapidfuzz not installed — return dummy always returning False
        return lambda tokens: False


# ╔══════════════════════════════════════════════════════════════╗
# ║   CORE MATCHING  — applies all 3 layers to one text         ║
# ╚══════════════════════════════════════════════════════════════╝

def match_one_text(text, expanded_benefits, expanded_side_effects, fuzzy_fns):
    """
    Check a single text against all aspects using 3 layers.
    Returns a dict of {aspect_name: 0_or_1} plus summary columns.
    """
    # Empty text guard
    if not isinstance(text, str) or not text.strip():
        result = {asp: 0 for asp in ALL_ASPECTS}
        result.update({
            "benefit_count": 0, "side_effect_count": 0,
            "benefit_aspects": "", "side_effect_aspects": "",
            "has_benefit": 0, "has_side_effect": 0, "has_both": 0,
        })
        return result

    text_lower = text.lower()
    tokens     = re.findall(r"[a-z]{3,}", text_lower)   # for fuzzy layer

    found_ben = []
    found_se  = []
    result    = {}

    def check_aspect(aspect, keywords):
        # Layer 1 + 2: direct string match (seed + expanded)
        for kw in keywords:
            if kw in text_lower:
                return True
        # Layer 3: fuzzy match
        if aspect in fuzzy_fns:
            return fuzzy_fns[aspect](tokens)
        return False

    for aspect, keywords in expanded_benefits.items():
        hit = check_aspect(aspect, keywords)
        result[aspect] = 1 if hit else 0
        if hit:
            found_ben.append(aspect)

    for aspect, keywords in expanded_side_effects.items():
        hit = check_aspect(aspect, keywords)
        result[aspect] = 1 if hit else 0
        if hit:
            found_se.append(aspect)

    # Summary columns — used directly in later analyses
    result["benefit_count"]      = len(found_ben)
    result["side_effect_count"]  = len(found_se)
    result["benefit_aspects"]    = "|".join(found_ben)
    result["side_effect_aspects"]= "|".join(found_se)
    result["has_benefit"]        = 1 if found_ben else 0
    result["has_side_effect"]    = 1 if found_se  else 0
    result["has_both"]           = 1 if (found_ben and found_se) else 0

    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║   FREQUENCY TABLE                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def make_frequency_table(df):
    total = len(df)
    rows  = []

    for aspect in list(BENEFITS.keys()) + list(SIDE_EFFECTS.keys()):
        if aspect not in df.columns:
            continue
        n   = int(df[aspect].sum())
        cat = "Benefit" if aspect in BENEFITS else "Side Effect"
        rows.append({
            "aspect"   : aspect,
            "category" : cat,
            "count"    : n,
            "percent"  : round(n / total * 100, 2) if total > 0 else 0.0,
        })

    freq = pd.DataFrame(rows).sort_values("count", ascending=False)
    freq["rank"] = range(1, len(freq) + 1)
    return freq


# ╔══════════════════════════════════════════════════════════════╗
# ║   PLAIN ENGLISH REPORT                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def write_report(df, freq_df, path):
    total = len(df)
    n_b   = int(df["has_benefit"].sum())
    n_se  = int(df["has_side_effect"].sum())
    n_bo  = int(df["has_both"].sum())

    lines = []
    lines.append("=" * 62)
    lines.append("  ANALYSIS 1: ASPECT EXTRACTION — RESULTS")
    lines.append("  Ozempic Consumer Discourse Study")
    lines.append("=" * 62)

    lines.append(f"\n  DATASET OVERVIEW")
    lines.append(f"  Total records analysed : {total}")
    if "source_type" in df.columns:
        for src, n in df["source_type"].value_counts().items():
            lines.append(f"  {src:<28}: {n}  ({n/total*100:.1f}%)")

    lines.append(f"\n  COVERAGE")
    lines.append(f"  Posts with ≥1 benefit      : {n_b}  ({n_b/total*100:.1f}%)")
    lines.append(f"  Posts with ≥1 side effect  : {n_se}  ({n_se/total*100:.1f}%)")
    lines.append(f"  Posts mentioning BOTH       : {n_bo}  ({n_bo/total*100:.1f}%)")
    lines.append(f"  → These {n_bo} posts are the core trade-off corpus for Analysis 2")

    lines.append(f"\n  TOP BENEFITS:")
    for _, row in freq_df[freq_df["category"]=="Benefit"].head(5).iterrows():
        bar = "█" * max(1, int(row["percent"] / 2))
        lines.append(f"    {row['aspect']:<28} {bar}  {row['percent']:.1f}%  (n={row['count']})")

    lines.append(f"\n  TOP SIDE EFFECTS:")
    for _, row in freq_df[freq_df["category"]=="Side Effect"].head(5).iterrows():
        bar = "█" * max(1, int(row["percent"] / 2))
        lines.append(f"    {row['aspect']:<28} {bar}  {row['percent']:.1f}%  (n={row['count']})")

    lines.append(f"\n  INTERPRETATION FOR RO1:")
    lines.append(f"  {n_bo/total*100:.1f}% of records simultaneously mention benefits")
    lines.append(f"  and side effects — confirming that dual-evaluation discourse")
    lines.append(f"  exists and providing the aspect-level labels needed for")
    lines.append(f"  Analysis 2 (ABSA) and Analysis 6 (trade-off detection).")

    lines.append(f"\n  OUTPUT FILES:")
    lines.append(f"  results/A1_aspect_results.csv   ← full dataset with flags")
    lines.append(f"  results/A1_frequency_table.csv  ← aspect frequency counts")
    lines.append(f"  results/A1_report.txt           ← this report")
    lines.append("=" * 62)

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ╔══════════════════════════════════════════════════════════════╗
# ║   MAIN PIPELINE                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  ANALYSIS 1 — ASPECT EXTRACTION")
    print("  3-Layer: Dictionary + FastText + Fuzzy Matching")
    print("=" * 62)

    # ── Step 1: Load data ────────────────────────────────────
    print("\n  [Step 1] Loading data...")

    dfs = []

    if os.path.exists(PRIMARY_FILE):
        df_primary = pd.read_csv(PRIMARY_FILE, encoding="utf-8-sig")
        df_primary["source_type"] = "Reddit"
        dfs.append(df_primary)
        print(f"  Primary   : {len(df_primary):>5} rows  ({PRIMARY_FILE})")
    else:
        print(f"  ⚠  Primary file not found: {PRIMARY_FILE}")
        print(f"     Run scraper_primary_full.py  (full scale)")
        print(f"     OR   scraper_primary.py      (pilot 1,000 rows)")

    if os.path.exists(SECONDARY_FILE):
        df_secondary = pd.read_csv(SECONDARY_FILE, encoding="utf-8-sig")
        df_secondary["source_type"] = "Review_Platform"
        dfs.append(df_secondary)
        print(f"  Secondary : {len(df_secondary):>5} rows  ({SECONDARY_FILE})")
    else:
        print(f"  ⚠  Secondary file not found: {SECONDARY_FILE}")
        print(f"     Run scraper_secondary_full.py  (full scale)")
        print(f"     OR   scraper_secondary.py      (pilot 1,000 rows)")

    if not dfs:
        print("\n  ✗ No data files found. Run both scrapers first.")
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Ensure combined_text column exists
    if TEXT_COLUMN not in df.columns:
        # Try to build it from whatever text columns exist
        text_cols = [c for c in ["combined_text", "text", "post_text",
                                  "review_text", "body"] if c in df.columns]
        if text_cols:
            df[TEXT_COLUMN] = df[text_cols[0]].fillna("")
            print(f"  Using '{text_cols[0]}' as text column")
        else:
            print(f"  ✗ No text column found. Available: {list(df.columns)}")
            return None

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    texts = df[TEXT_COLUMN].tolist()
    print(f"\n  Total records to analyse: {len(df)}")

    # ── Step 2: FastText expansion (Layer 2) ─────────────────
    print(f"\n  [Step 2] FastText Semantic Expansion")
    exp_benefits   = dict(BENEFITS)
    exp_side_effects = dict(SIDE_EFFECTS)

    if len(texts) >= FASTTEXT_MIN_CORPUS:
        print(f"  Corpus size {len(texts):,} ≥ threshold {FASTTEXT_MIN_CORPUS:,} → FastText activating")
        ft = train_fasttext(texts)
        if ft:
            print("  Expanding benefit aspects...")
            exp_benefits = expand_with_fasttext(
                ft, BENEFITS, FASTTEXT_SIMILARITY, FASTTEXT_TOP_N)
            print("  Expanding side-effect aspects...")
            exp_side_effects = expand_with_fasttext(
                ft, SIDE_EFFECTS, FASTTEXT_SIMILARITY, FASTTEXT_TOP_N)

            # Save expanded dictionary for transparency / supervisor review
            all_exp = {**exp_benefits, **exp_side_effects}
            dict_path = os.path.join(OUTPUT_DIR, "A1_expanded_dictionary.json")
            with open(dict_path, "w") as f:
                json.dump(all_exp, f, indent=2)
            print(f"  Expanded dictionary saved → {dict_path}")
        else:
            print("  Using seed dictionary only (FastText unavailable)")
    else:
        print(f"  Corpus size {len(texts)} < threshold {FASTTEXT_MIN_CORPUS}")
        print("  Using seed dictionary only — this is fine for a pilot dataset")

    # ── Step 3: Build fuzzy matchers (Layer 3) ───────────────
    print(f"\n  [Step 3] Building Fuzzy Matchers (Layer 3)")
    fuzzy_fns = {}
    has_fuzzy = False
    try:
        import rapidfuzz  # noqa — just checking availability
        for aspect, kws in {**exp_benefits, **exp_side_effects}.items():
            fuzzy_fns[aspect] = make_fuzzy_checker(kws)
        has_fuzzy = True
        print(f"  ✅ Fuzzy matchers ready for {len(fuzzy_fns)} aspects")
    except ImportError:
        print("  ⚠  rapidfuzz not installed — fuzzy matching skipped")
        print("     Install: pip install rapidfuzz")

    # ── Step 4: Match aspects in all records ─────────────────
    print(f"\n  [Step 4] Extracting aspects from {len(df)} records...")
    aspect_rows = []
    for i, text in enumerate(texts):
        aspect_rows.append(
            match_one_text(text, exp_benefits, exp_side_effects, fuzzy_fns))
        if (i + 1) % 200 == 0:
            print(f"    {i+1} / {len(texts)} processed", end="\r")

    print(f"    {len(texts)} / {len(texts)} ✅ done        ")

    aspects_df = pd.DataFrame(aspect_rows)
    df_out     = pd.concat([df.reset_index(drop=True), aspects_df], axis=1)

    # ── Step 5: Frequency table ───────────────────────────────
    print(f"\n  [Step 5] Computing aspect frequencies...")
    freq_df = make_frequency_table(df_out)

    # ── Step 6: Save outputs ──────────────────────────────────
    print(f"\n  [Step 6] Saving results...")

    path_results = os.path.join(OUTPUT_DIR, "A1_aspect_results.csv")
    path_freq    = os.path.join(OUTPUT_DIR, "A1_frequency_table.csv")
    path_report  = os.path.join(OUTPUT_DIR, "A1_report.txt")

    df_out.to_csv(path_results, index=False, encoding="utf-8-sig")
    freq_df.to_csv(path_freq, index=False, encoding="utf-8-sig")
    report_text = write_report(df_out, freq_df, path_report)

    # ── Print report ──────────────────────────────────────────
    print("\n" + report_text)

    print(f"\n  Output files saved in  results/")
    print(f"  ├── A1_aspect_results.csv     ← pass this to Analysis 2")
    print(f"  ├── A1_frequency_table.csv    ← aspect counts for write-up")
    if has_fuzzy and len(texts) >= FASTTEXT_MIN_CORPUS:
        print(f"  ├── A1_expanded_dictionary.json ← shows what FastText discovered")
    print(f"  └── A1_report.txt             ← plain English summary")

    return df_out, freq_df


if __name__ == "__main__":
    run()