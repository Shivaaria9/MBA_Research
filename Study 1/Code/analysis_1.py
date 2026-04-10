import pandas as pd
import numpy as np
import re, os, json
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Optional imports (graceful fallback if not installed) ─────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    USE_SPACY = True
except Exception:
    USE_SPACY = False
    print("⚠ spaCy not found — install: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    USE_SPELLCHECK = True
except Exception:
    USE_SPELLCHECK = False
    print("⚠ pyspellchecker not found — install: pip install pyspellchecker")

try:
    from gensim.models import FastText
    USE_FASTTEXT = True
except Exception:
    USE_FASTTEXT = False
    print("⚠ gensim not found — install: pip install gensim")

import nltk
for r in ["punkt", "stopwords", "wordnet"]:
    try: nltk.data.find(f"tokenizers/{r}")
    except: nltk.download(r, quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════════
# LAYER 1: TEXT NORMALIZATION
# Handles abbreviations, slang, formatting issues in Reddit text
# ═══════════════════════════════════════════════════════════════

# Medical / Ozempic abbreviations — PRESERVE these
MEDICAL_KEEP = {
    "a1c", "hba1c", "t2d", "gi", "bp", "bmi",
    "ozempic", "semaglutide", "tirzepatide", "mounjaro",
    "glp1", "glp-1", "t2", "lbs", "kg",
}

# Reddit abbreviations → expand
ABBREVIATION_MAP = {
    # Weight / measurement
    r'\b(\d+)lbs\b':      r'\1 lbs',
    r'\b(\d+)lb\b':       r'\1 lb',
    r'\b(\d+)kgs?\b':     r'\1 kg',
    r'\b(\d+)wks?\b':     r'\1 weeks',
    r'\b(\d+)mos?\b':     r'\1 months',
    # Common Reddit shorthand
    r'\bbc\b':            'because',
    r'\btbh\b':           '',
    r'\blol\b':           '',
    r'\blmao\b':          '',
    r'\bfr\b':            '',
    r'\brn\b':            '',
    r'\bomg\b':           '',
    r'\bngl\b':           '',
    r'\bimo\b':           '',
    r'\bbtw\b':           'by the way',
    r'\bidk\b':           'i do not know',
    r'\bngl\b':           '',
    r'\bwut\b':           'what',
    r'\by\'all\b':        'everyone',
    r'\bgonna\b':         'going to',
    r'\bwanna\b':         'want to',
    r'\bgotta\b':         'got to',
    r'\bkinda\b':         'kind of',
    r'\bsorta\b':         'sort of',
    r'\bpretty much\b':   'mostly',
    # Number + unit patterns
    r'\b(\d+)\s*(?:pound|lb)s?\b': r'\1 lbs',
    r'\bdown\s+(\d+)\b':           r'lost \1',
    r'-(\d+)\s*lbs?\b':            r'lost \1 lbs',
}

# Reddit slang → standard terms (for matching purposes only, not for display)
SLANG_NORMALIZATION = {
    # Nausea synonyms
    "hurling":        "vomiting",
    "hurled":         "vomiting",
    "hurl":           "vomiting",
    "barfed":         "vomited",
    "barfing":        "vomiting",
    "barf":           "vomit",
    "puking":         "vomiting",
    "puked":          "vomited",
    "puke":           "vomit",
    "threw up":       "vomited",
    "throw up":       "vomit",
    "heaving":        "vomiting",
    "feeling sick":   "nausea",
    "feel sick":      "nausea",
    "queasy":         "nausea",
    "nausious":       "nauseous",
    # GI synonyms
    "the runs":       "diarrhea",
    "loose stool":    "diarrhea",
    "stomach crzy":   "stomach issues",
    "bathroom trips": "diarrhea",
    "toilet trips":   "diarrhea",
    "gi stuff":       "gastrointestinal issues",
    "tummy":          "stomach",
    "guts":           "stomach",
    # Fatigue
    "zombie":         "exhausted",
    "wiped out":      "exhausted",
    "burnt out":      "fatigued",
    "drained":        "fatigue",
    # Ozempic community terms (do NOT normalize — add to dictionary instead)
    # "ozempic face" — keep as-is, added to dictionary
    # "food noise"   — keep as-is, added to dictionary
    # "sulfur burps" — keep as-is, added to dictionary
    # "brain fog"    — keep as-is, added to dictionary
}

def normalize_text(text):
    """
    Layer 1: Normalize Reddit-specific abbreviations and formatting.

    What this handles:
      - URLs, subreddit/user mentions
      - Weight formats: "22lbs" → "22 lbs", "-3lbs" → "lost 3 lbs"
      - Reddit abbrevs: "tbh", "lol", "bc", "fr" etc.
      - Slang → standard equivalents (barfing → vomiting)

    What this does NOT change:
      - Medical abbreviations: a1c, hba1c, gi, t2d
      - Drug names: ozempic, semaglutide
      - Community terms: food noise, sulfur burps, ozempic face
        (these are added directly to dictionary)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove noise
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'/r/\w+|/u/\w+', '', text)
    text = re.sub(r'&amp;|&lt;|&gt;|&quot;|&#\d+;', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?,\'\-\%]', ' ', text)

    # Apply abbreviation expansions
    for pattern, replacement in ABBREVIATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Apply slang normalization
    for slang, standard in SLANG_NORMALIZATION.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', standard, text, flags=re.IGNORECASE)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# LAYER 2: SPELL CORRECTION
# Handles misspellings common in Reddit posts
# ═══════════════════════════════════════════════════════════════

# Medical terms to PROTECT from spell correction
# (spell checkers would incorrectly "fix" these)
PROTECTED_MEDICAL_TERMS = {
    "ozempic", "semaglutide", "tirzepatide", "mounjaro", "wegovy",
    "metformin", "insulin", "a1c", "hba1c", "glp1", "t2d",
    "endocrinologist", "prediabetes", "glycemic", "bariatric",
    # Community terms — do not correct
    "ozempic", "food noise", "sulfur burps", "ozempic face", "brain fog",
}

# Manual override corrections for health/Reddit context
MANUAL_CORRECTIONS = {
    # Nausea misspellings
    "nausia":        "nausea",
    "nausious":      "nauseous",
    "nausited":      "nauseated",
    "nasea":         "nausea",
    "naseua":        "nausea",
    # GI misspellings
    "constipashun":  "constipation",
    "constipatd":    "constipated",
    "diarreah":      "diarrhea",
    "diareah":       "diarrhea",
    "diarrhoea":     "diarrhea",
    "diarreha":      "diarrhea",
    "diahrea":       "diarrhea",
    # Fatigue misspellings
    "tierd":         "tired",
    "exausted":      "exhausted",
    "fatiuge":       "fatigue",
    "lethargic":     "lethargic",
    # Hair misspellings
    "thinning":      "thinning",
    "shedding":      "shedding",
    # General Reddit misspellings
    "loosing":       "losing",
    "loosing weight":"losing weight",
    "crzy":          "crazy",
    "wrk":           "work",
}

def correct_spelling(text):
    """
    Layer 2: Correct misspellings using manual corrections + pyspellchecker.

    Strategy:
    1. Apply manual corrections first (highest precision for health terms)
    2. Use pyspellchecker for remaining unknown words
    3. ALWAYS protect medical/drug terms from correction

    Note: Spell correction is applied WORD by WORD — short words (≤3 chars)
    are skipped to avoid over-correction.
    """
    if not text:
        return text

    # Step 1: Apply manual corrections (highest priority)
    for wrong, right in MANUAL_CORRECTIONS.items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text, flags=re.IGNORECASE)

    # Step 2: pyspellchecker (if available)
    if USE_SPELLCHECK:
        words = text.split()
        corrected = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word).lower()
            # Skip: protected terms, short words, numbers, already-clean
            if (clean_word in PROTECTED_MEDICAL_TERMS
                    or len(clean_word) <= 3
                    or clean_word.isdigit()):
                corrected.append(word)
                continue
            # Get suggestion
            correction = spell.correction(clean_word)
            if correction and correction != clean_word:
                corrected.append(word.replace(clean_word, correction))
            else:
                corrected.append(word)
        text = " ".join(corrected)

    return text


# ═══════════════════════════════════════════════════════════════
# LAYER 3: LEMMATIZATION
# ═══════════════════════════════════════════════════════════════

def lemmatize_text(text):
    """
    Layer 3: Lemmatize using spaCy (preferred) or NLTK fallback.
    Handles: losing→lose, dropped→drop, pounds→pound, nauseous→nauseou (spaCy)
    """
    if not text:
        return ""
    if USE_SPACY:
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_space])
    else:
        words = text.lower().split()
        return " ".join([lemmatizer.lemmatize(w, pos='v') for w in words])


# ═══════════════════════════════════════════════════════════════
# LAYER 4A: EXPANDED DICTIONARY
# Includes slang, community terms, abbreviations — all verified
# ═══════════════════════════════════════════════════════════════

ASPECTS = {

    # ══ BENEFITS ══════════════════════════════════════════════

    "weight_loss": {
        "category": "benefit",
        "description": "Weight reduction",
        "keywords": [
            # Standard terms
            "weight loss", "weight", "pounds", "lbs", "lost", "loss",
            "drop", "dropped", "shed", "slim", "lighter", "scale",
            "kilos", "kg",
            # Reddit formats (after normalization: "22lbs" → "22 lbs")
            "lbs down", "pounds down", "pounds gone", "lbs gone",
            "losing weight", "lost weight", "losing lbs",
            # Informal expressions found on real Reddit
            "melting off", "weight melting", "the weight is coming off",
            "dress sizes", "clothes fitting", "fitting into",
            "scale going down", "scale dropping", "scale shows",
            "shed the weight", "shed pounds", "dropping pounds",
            "pounds lighter", "lighter on the scale",
        ]
    },

    "appetite_suppression": {
        "category": "benefit",
        "description": "Reduced hunger and food cravings",
        "keywords": [
            "appetite", "hunger", "hungry", "craving", "cravings",
            "eating less", "not hungry", "full", "satiety", "satisfied",
            # Reddit community-coined terms (high frequency in real data)
            "food noise",        # 120 occ in GenAI; higher in real data
            "thinking about food", "food 24/7", "not thinking about food",
            "forget to eat", "forgot to eat", "no interest in food",
            "food cravings gone", "no cravings",
            "snacking", "binge", "portion", "portions",
            "reduced appetite", "appetite gone", "appetite suppressed",
            # Informal Reddit phrases
            "could eat forever", "always thinking about food",
            "food thoughts", "obsessed with food",
        ]
    },

    "glucose_control": {
        "category": "benefit",
        "description": "Blood glucose and diabetes management",
        "keywords": [
            "glucose", "blood sugar", "a1c", "hba1c", "insulin",
            "diabetes", "diabetic", "sugar levels", "glycemic",
            "endocrinologist", "readings", "normal range",
            "blood glucose", "prediabetes", "type 2", "t2d",
            "sugar control", "sugar stable", "glucose readings",
            # Reddit formats
            "a1c dropped", "a1c went from", "a1c went down",
            "blood sugar down", "sugar down", "numbers down",
            "no more spikes", "sugar spikes gone",
        ]
    },

    "energy_improvement": {
        "category": "benefit",
        "description": "Improved energy and vitality",
        "keywords": [
            "energy", "energetic", "active", "stamina",
            "less tired", "more energy", "exhaustion gone",
            "energy levels", "through the roof",
            # Reddit informal
            "so much energy", "tons of energy", "energy back",
            "not dragging", "not exhausted anymore",
        ]
    },

    "confidence_wellbeing": {
        "category": "benefit",
        "description": "Psychological wellbeing and confidence",
        "keywords": [
            "confidence", "self-esteem", "body image", "proud",
            "transformation", "new me", "feel better", "motivated",
            "positive", "healthier", "different person", "surreal",
            "life-changing", "life changing", "game changer",
            # Real Reddit expressions
            "people are noticing", "people are starting to notice",
            "compliments", "feel amazing", "feel incredible",
            "never felt better", "best i have ever felt",
            "look in the mirror", "love my body",
            "clothes fit", "fitting clothes",
        ]
    },

    "cardiovascular_health": {
        "category": "benefit",
        "description": "Heart and cardiovascular improvements",
        "keywords": [
            "blood pressure", "cholesterol", "triglycerides",
            "bp", "hypertension", "lipids", "heart health",
            "cardiovascular", "health markers", "labs improved",
            "numbers improved", "bloodwork", "blood work",
        ]
    },

    # ══ SIDE EFFECTS ══════════════════════════════════════════

    "nausea": {
        "category": "side_effect",
        "description": "Nausea and vomiting",
        "keywords": [
            # Standard terms
            "nausea", "nauseous", "vomiting", "vomit",
            "throwing up", "thrown up", "sick",
            # Real Reddit slang (after SLANG_NORMALIZATION these become standard)
            # but also keep originals in case normalization misses any
            "barfing", "barf", "barfed",
            "puking", "puke", "puked",
            "throwing up", "threw up",
            "hurling", "hurl", "hurled",
            "queasy",                       # correctly spelled informal term
            # Community expressions
            "stomach sick", "sick to stomach", "feel like vomiting",
            "waves of nausea", "nauseated", "debilitating nausea",
            "morning sickness", "nausea waves",
            # Reddit descriptions
            "stomach doing flips", "stomach churning",
            "want to puke", "about to puke",
        ]
    },

    "gastrointestinal": {
        "category": "side_effect",
        "description": "Digestive and GI issues",
        "keywords": [
            "diarrhea", "constipation", "stomach", "digestive",
            "bloating", "bloated", "gas", "cramps", "bowel",
            "heartburn", "acid reflux", "acidic",
            "digestive slowdown", "gi issues", "gi stuff",
            # Reddit community-coined term (very common in real data)
            "sulfur burps",          # Reddit-specific Ozempic term
            "sulfur burp",
            "sulfur belching",
            "egg burps",             # alternative name used on Reddit
            # Informal expressions
            "bathroom trips", "bathroom issues", "running to bathroom",
            "bathroom every hour", "toilet issues",
            "stomach issues", "tummy issues",
            "stomach going crazy", "stomach cramps",
            "loose stool", "loose stools", "the runs",
            "bowel issues", "gut issues", "gut problems",
            "indigestion", "upset stomach",
        ]
    },

    "fatigue": {
        "category": "side_effect",
        "description": "Fatigue and tiredness",
        "keywords": [
            "fatigue", "tired", "exhausted", "tiredness",
            "lethargy", "drained", "no energy", "sluggish",
            "exhaustion", "lethargic",
            # Reddit informal
            "wiped out", "burnt out", "feel like a zombie",
            "zombie mode", "can barely function",
            "hitting a wall", "energy crash",
            "always tired", "so tired", "extremely tired",
            "fatigue hits", "fatigue in waves",
        ]
    },

    "headaches": {
        "category": "side_effect",
        "description": "Headaches and migraines",
        "keywords": [
            "headache", "headaches", "migraine", "head pain",
            "splitting headache", "persistent headaches",
            # Reddit informal
            "head is pounding", "pounding headache",
            "head splitting", "killer headache",
            "head hurts", "my head", "constant headaches",
        ]
    },

    "injection_reactions": {
        "category": "side_effect",
        "description": "Injection site reactions",
        "keywords": [
            "injection", "inject", "needle", "pen",
            "dose", "dosage", "shot", "injection site",
            "site reaction", "bruise", "redness", "bump",
            "lump", "swelling",
            # Reddit informal
            "shooting up", "jab", "stabbing myself",
            "pen hurts", "painful injection",
        ]
    },

    "hair_loss": {
        "category": "side_effect",
        "description": "Hair thinning or loss",
        "keywords": [
            "hair loss", "hair thinning", "thinning",
            "hair shedding", "losing hair", "hair falling",
            "shedding", "bald",
            # Reddit expressions
            "hair falling out", "clumps of hair",
            "hair in shower", "hair everywhere",
            "hair is gone", "ozempic hair",
        ]
    },

    "ozempic_face": {
        "category": "side_effect",
        "description": "Facial volume loss (community-coined term)",
        "keywords": [
            # Reddit-coined community term — not in any medical dictionary
            "ozempic face",
            "face sagging", "face gaunt", "gaunt face",
            "face changes", "face looking old",
            "face volume", "lost volume in face",
            "older looking", "face aging",
            "saggy face", "face looks different",
            "facial wasting", "face too thin",
        ]
    },

    "brain_fog": {
        "category": "side_effect",
        "description": "Cognitive effects and mental clarity issues",
        "keywords": [
            # Community-coined term
            "brain fog",
            "foggy brain", "foggy thinking",
            # Reddit informal
            "cant concentrate", "cannot concentrate",
            "hard to think", "trouble thinking",
            "memory issues", "forgetting things",
            "mental clarity", "cognitive", "brain not working",
            "not thinking clearly", "confused",
            "trouble focusing", "hard to focus",
        ]
    },

    "mental_effects": {
        "category": "side_effect",
        "description": "Mood and psychological side effects",
        "keywords": [
            "depression", "anxiety", "mood", "mood swings",
            "irritable", "emotional", "mental health",
            "psychological", "worried", "panic",
            # Reddit expressions
            "feeling down", "feeling low", "crying",
            "emotional wreck", "mood is off",
            "not myself", "feel different mentally",
        ]
    },
}


# ═══════════════════════════════════════════════════════════════
# LAYER 4B: OPTIONAL FASTTEXT SEMANTIC EXPANSION
# Use this when you have 10,000+ posts (full scraped dataset)
# ═══════════════════════════════════════════════════════════════

def train_fasttext_and_expand(df, aspects, min_corpus_size=1000,
                               similarity_threshold=0.62, topn=12):
    """
    Optional Layer 4B: Train FastText on scraped corpus and
    expand dictionary with discovered synonyms.

    When to use:
      ✓ You have 10,000+ posts from real scraped data
      ✓ Vocabulary diversity is high (Reddit slang heavy)
      ✗ Skip for GenAI/structured data (insufficient vocabulary)
      ✗ Skip if corpus < 1,000 texts (embeddings unreliable)

    FastText advantages over Word2Vec for Reddit:
      - Handles "nausia" → similar vector to "nausea" via char n-grams
      - Handles "barfing/barfed" → same concept automatically
      - Works on out-of-vocabulary words (sub-word model)

    Parameters:
      similarity_threshold: 0.62 (lower than typical 0.65 for Reddit
                            since informal text has noisier vectors)
      topn: 12 nearest neighbours per seed word
    """
    if not USE_FASTTEXT:
        print("⚠ FastText not available — skipping semantic expansion")
        return aspects

    corpus_size = len(df)
    if corpus_size < min_corpus_size:
        print(f"⚠ Corpus too small ({corpus_size} posts) for reliable FastText.")
        print(f"  Need {min_corpus_size}+ posts. Skipping expansion.")
        return aspects

    print(f"\n─── FASTTEXT SEMANTIC EXPANSION ─────────────────────────")
    print(f"  Corpus size: {corpus_size:,} posts — training FastText...")

    # Tokenize preprocessed text
    sentences = [text.split() for text in df["text_processed"].tolist()
                 if isinstance(text, str) and len(text.split()) > 3]

    # Train FastText (subword model — handles misspellings)
    model = FastText(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,          # Word must appear 3+ times
        workers=4,
        epochs=10,
        sg=1,                 # Skip-gram (better for rare/informal words)
        min_n=3,              # Min char n-gram (typo handling)
        max_n=6,
    )

    print(f"  Vocabulary trained: {len(model.wv):,} tokens")

    expanded_aspects = {}
    for aspect_name, aspect_data in aspects.items():
        seeds = aspect_data["keywords"]
        discovered = []
        all_terms = set(kw.lower() for kw in seeds)

        for seed in seeds[:8]:  # Use first 8 seeds per aspect
            seed_lower = seed.lower()
            try:
                similar = model.wv.most_similar(seed_lower, topn=topn)
                for word, score in similar:
                    if (score >= similarity_threshold
                            and word not in all_terms
                            and len(word) > 3
                            and not word.isdigit()):
                        all_terms.add(word)
                        discovered.append(word)
            except KeyError:
                pass  # Seed not in vocabulary

        expanded_aspects[aspect_name] = {
            **aspect_data,
            "keywords": list(set(seeds) | all_terms),
            "keywords_discovered": discovered,
        }
        if discovered:
            print(f"  [{aspect_name}] +{len(discovered)} terms: {discovered[:5]}")

    return expanded_aspects


# ═══════════════════════════════════════════════════════════════
# FULL PREPROCESSING PIPELINE (All 4 Layers Combined)
# ═══════════════════════════════════════════════════════════════

def full_preprocessing_pipeline(text):
    """
    Apply all 4 layers in sequence:

    Input:  "nausia is killing me rn. tried ginger tea. down 22lbs tho"
    Layer1: "nausea is killing me. tried ginger tea. lost 22 lbs though"
    Layer2: "nausea is killing me. tried ginger tea. lost 22 lbs though"
    Layer3: "nausea be kill me. try ginger tea. lose 22 lb though"
    Output: ready for dictionary matching

    Note: we match against BOTH original text and preprocessed text
    for maximum coverage.
    """
    if not isinstance(text, str):
        return "", ""

    original = text                          # Keep original for evidence reporting
    normalized = normalize_text(text)        # Layer 1
    spell_corrected = correct_spelling(normalized)  # Layer 2
    lemmatized = lemmatize_text(spell_corrected)    # Layer 3
    # Layer 4 (dictionary matching) happens in detect_aspects_in_text()

    return spell_corrected, lemmatized


def build_lemmatized_dictionary(aspects):
    """Pre-lemmatize all dictionary keywords for fast matching."""
    print("─── Building lemmatized dictionary ─────────────────────────")
    lemma_aspects = {}
    for aspect_name, aspect_data in aspects.items():
        original_kws = aspect_data["keywords"]
        all_kws = set()
        for kw in original_kws:
            all_kws.add(kw.lower())
            all_kws.add(lemmatize_text(kw.lower()))
            # Also add normalized version
            all_kws.add(normalize_text(kw.lower()))
        lemma_aspects[aspect_name] = {
            **aspect_data,
            "keywords_all": list(all_kws),
        }
        print(f"  [{aspect_name}]  {len(original_kws)} original → {len(all_kws)} (with lemmas)")
    return lemma_aspects


# ═══════════════════════════════════════════════════════════════
# ASPECT DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════

def detect_aspects_in_text(original_text, lemma_aspects):
    """
    Detect aspects using expanded + lemmatized dictionary.
    Matches against BOTH:
      - spell-corrected + normalized text (catches misspellings)
      - lemmatized text (catches morphological variants)
      - original text (safety net for any missed normalizations)
    """
    if not isinstance(original_text, str) or len(original_text.strip()) == 0:
        return {a: {"detected": False, "evidence": [], "category": d["category"]}
                for a, d in lemma_aspects.items()}

    # Prepare all text versions for matching
    spell_corrected, lemmatized = full_preprocessing_pipeline(original_text)
    text_original_lower = original_text.lower()

    # All versions to match against
    text_versions = [text_original_lower, spell_corrected, lemmatized]

    results = {}
    for aspect_name, aspect_data in lemma_aspects.items():
        evidence = []

        for kw in aspect_data["keywords_all"]:
            kw_lower = kw.lower()
            if len(kw_lower) < 3:
                continue
            # Check keyword against all text versions
            for tv in text_versions:
                if kw_lower in tv:
                    evidence.append(kw_lower)
                    break  # Found in at least one version

        results[aspect_name] = {
            "detected": len(evidence) > 0,
            "evidence": list(set(evidence))[:3],
            "category": aspect_data["category"],
        }

    return results


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(reddit_path=None, reviews_path=None):
    """
    Load data from CSV files.
    Handles both the current GenAI dataset and real scraped data.

    For real scraped data, expected columns:
      Reddit: post_id, post_text, comment_text (optional), subreddit, post_date
      Reviews: review_id, review_text, rating, platform, review_date
    """
    print("\n─── LOADING DATA ────────────────────────────────────────────")

    # Default paths
    if not reddit_path:
        for p in ["F:/MBA 2025-27/SEM 2/3_CREDIT/Research structure/Study 1/Study 1_Analysis 1/Dataset/primary_ozempic_reddit_data.csv",
                  "primary_ozempic_reddit_data.csv"]:
            if os.path.exists(p):
                reddit_path = p
                break

    if not reviews_path:
        for p in ["F:/MBA 2025-27/SEM 2/3_CREDIT/Research structure/Study 1/Study 1_Analysis 1/Dataset/secondary_ozempic_reviews_data.csv",
                  "secondary_ozempic_reviews_data.csv"]:
            if os.path.exists(p):
                reviews_path = p
                break

    dfs = []

    if reddit_path and os.path.exists(reddit_path):
        reddit = pd.read_csv(reddit_path)
        reddit["combined_text"] = (
            reddit.get("post_text", pd.Series([""] * len(reddit))).fillna("") + " " +
            reddit.get("comment_text", pd.Series([""] * len(reddit))).fillna("")
        ).str.strip()
        reddit["source"] = "Reddit"
        reddit["text_id"] = reddit.get("post_id", reddit.index.astype(str))
        reddit["platform"] = reddit.get("subreddit", "Reddit")
        dfs.append(reddit[["text_id", "source", "platform", "combined_text",
                            "sentiment_label",
                            "contains_benefit_mention",
                            "contains_side_effect_mention",
                            "contains_trade_off_language"]])
        print(f"✓ Reddit: {len(reddit):,} posts")

    if reviews_path and os.path.exists(reviews_path):
        reviews = pd.read_csv(reviews_path)
        reviews["combined_text"] = reviews.get("review_text", pd.Series([""] * len(reviews))).fillna("")
        reviews["source"] = "Review_Platform"
        reviews["text_id"] = reviews.get("review_id", reviews.index.astype(str))
        reviews["platform"] = reviews.get("platform", "Reviews")
        dfs.append(reviews[["text_id", "source", "platform", "combined_text",
                             "sentiment_label",
                             "contains_benefit_mention",
                             "contains_side_effect_mention",
                             "contains_trade_off_language"]])
        print(f"✓ Reviews: {len(reviews):,} reviews")

    if not dfs:
        raise FileNotFoundError("No data files found. Check file paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Combined: {len(df):,} total texts")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_aspect_extraction(df, lemma_aspects):
    """Run full aspect detection with preprocessing on all texts."""
    print("\n─── ASPECT EXTRACTION (with 4-layer preprocessing) ─────────")

    # Pre-process all texts (store for reuse in later analyses)
    tqdm.pandas(desc="Preprocessing texts")
    df["text_processed"] = df["combined_text"].progress_apply(
        lambda t: full_preprocessing_pipeline(t)[0]  # spell-corrected version
    )
    df["text_lemmatized"] = df["combined_text"].progress_apply(
        lambda t: full_preprocessing_pipeline(t)[1]  # lemmatized version
    )

    benefit_aspects     = [a for a, d in lemma_aspects.items() if d["category"] == "benefit"]
    side_effect_aspects = [a for a, d in lemma_aspects.items() if d["category"] == "side_effect"]

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting aspects"):
        detection = detect_aspects_in_text(row["combined_text"], lemma_aspects)

        record = {
            "text_id":                     row["text_id"],
            "source":                      row["source"],
            "platform":                    row["platform"],
            "sentiment_label":             row.get("sentiment_label", ""),
            "contains_benefit_mention":    row.get("contains_benefit_mention", 0),
            "contains_side_effect_mention": row.get("contains_side_effect_mention", 0),
            "contains_trade_off_language": row.get("contains_trade_off_language", 0),
            "text_preview":                str(row["combined_text"])[:120],
            "text_processed_preview":      str(row["text_processed"])[:120],
        }

        for aspect_name, result in detection.items():
            record[f"has_{aspect_name}"]      = int(result["detected"])
            record[f"evidence_{aspect_name}"] = "|".join(result["evidence"])

        record["benefit_count"]     = sum(int(detection[a]["detected"]) for a in benefit_aspects)
        record["side_effect_count"] = sum(int(detection[a]["detected"]) for a in side_effect_aspects)
        record["total_aspects"]     = record["benefit_count"] + record["side_effect_count"]
        record["has_both"]          = int(record["benefit_count"] > 0 and record["side_effect_count"] > 0)

        records.append(record)

    df_out = pd.DataFrame(records)

    # ── Coverage Report ───────────────────────────────────────
    total    = len(df_out)
    detected = (df_out["total_aspects"] > 0).sum()
    both     = df_out["has_both"].sum()

    print(f"\n  Coverage:              {detected}/{total} ({detected/total*100:.1f}%)")
    print(f"  Both (trade-off):      {both} ({both/total*100:.1f}%)")
    print(f"  Benefits only:         {((df_out['benefit_count']>0)&(df_out['side_effect_count']==0)).sum()}")
    print(f"  Side effects only:     {((df_out['side_effect_count']>0)&(df_out['benefit_count']==0)).sum()}")
    print(f"  Neither:               {(df_out['total_aspects']==0).sum()}")

    return df_out


# ═══════════════════════════════════════════════════════════════
# FREQUENCY, CO-OCCURRENCE, VISUALIZATION, SAVE
# (Same as before — see Analysis_1_Final.py for full versions)
# ═══════════════════════════════════════════════════════════════

def aspect_frequency_analysis(df_out, lemma_aspects):
    rows = []
    for aspect_name, aspect_data in lemma_aspects.items():
        col = f"has_{aspect_name}"
        count_total  = df_out[col].sum()
        count_reddit = df_out[df_out["source"] == "Reddit"][col].sum()
        count_review = df_out[df_out["source"] == "Review_Platform"][col].sum()
        rows.append({
            "aspect":       aspect_name.replace("_", " ").title(),
            "category":     aspect_data["category"],
            "total_count":  int(count_total),
            "pct_total":    round(count_total / len(df_out) * 100, 1),
            "reddit_count": int(count_reddit),
            "review_count": int(count_review),
        })
    df_freq = pd.DataFrame(rows).sort_values("total_count", ascending=False)
    print("\n─── ASPECT FREQUENCY ────────────────────────────────────────")
    print(df_freq.to_string(index=False))
    return df_freq


def save_outputs(df_out, df_freq, lemma_aspects):
    os.makedirs("outputs", exist_ok=True)
    df_out.to_csv("outputs/analysis1_full_results.csv", index=False)
    df_freq.to_csv("outputs/analysis1_frequency_summary.csv", index=False)
    trade_off = df_out[df_out["has_both"] == 1]
    trade_off.to_csv("outputs/analysis1_tradeoff_candidates.csv", index=False)
    print(f"\n✓ Saved: {len(df_out):,} rows to outputs/analysis1_full_results.csv")
    print(f"✓ Saved: {len(trade_off):,} trade-off posts to outputs/analysis1_tradeoff_candidates.csv")
    print(f"  → These trade-off posts feed directly into Analysis 2 (ABSA)")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_analysis1(reddit_path=None, reviews_path=None,
                  use_fasttext_expansion=False):
    """
    MAIN ENTRY POINT

    For current GenAI dataset:
      run_analysis1()  — uses uploaded CSV paths automatically

    For real scraped Reddit data:
      run_analysis1(
          reddit_path="scraped_reddit_ozempic.csv",
          reviews_path="scraped_reviews.csv",
          use_fasttext_expansion=True   # Enable when corpus >= 10,000 posts
      )
    """
    print("=" * 62)
    print("  ANALYSIS 1: ASPECT EXTRACTION — PRODUCTION VERSION")
    print("  4-Layer Pipeline: Normalize → Spell → Lemmatize → Dict")
    print("=" * 62)

    # Build dictionary
    lemma_aspects = build_lemmatized_dictionary(ASPECTS)

    # Load data
    df = load_data(reddit_path, reviews_path)

    # Optional FastText expansion (for large real scraped corpus)
    if use_fasttext_expansion and USE_FASTTEXT:
        # Pre-process texts first (FastText trains on clean text)
        df["text_processed"] = df["combined_text"].apply(
            lambda t: full_preprocessing_pipeline(t)[0])
        expanded = train_fasttext_and_expand(
            df, ASPECTS,
            min_corpus_size=1000,
            similarity_threshold=0.62,
            topn=12
        )
        lemma_aspects = build_lemmatized_dictionary(expanded)

    # Run extraction
    df_out = run_aspect_extraction(df, lemma_aspects)

    # Frequency analysis
    df_freq = aspect_frequency_analysis(df_out, lemma_aspects)

    # Save
    save_outputs(df_out, df_freq, lemma_aspects)

    print("\n" + "=" * 62)
    print("  ANALYSIS 1 COMPLETE")
    print(f"  Coverage:        {(df_out['total_aspects']>0).mean()*100:.1f}%")
    print(f"  Trade-off posts: {df_out['has_both'].sum()} ({df_out['has_both'].mean()*100:.1f}%)")
    print("=" * 62)

    return df_out, lemma_aspects


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── For GenAI/current dataset ────────────────────────────
    df_out, lemma_aspects = run_analysis1()

    # ── For real scraped Reddit data ─────────────────────────
    # df_out, lemma_aspects = run_analysis1(
    #     reddit_path="scraped_reddit_ozempic.csv",
    #     reviews_path="scraped_drug_reviews.csv",
    #     use_fasttext_expansion=True  # Enable for 10,000+ posts
    # )