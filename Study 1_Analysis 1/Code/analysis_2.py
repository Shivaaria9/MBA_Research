import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# NLP
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

# Try to load transformer model (optional)
TRANSFORMER_AVAILABLE = False  # Initialize first
try:
    from transformers import pipeline
    TRANSFORMER_AVAILABLE = True
    print("✓ Transformer models available")
except:
    TRANSFORMER_AVAILABLE = False
    print("⚠ Transformer models not available (using VADER only)")

# ============================================================================
# STEP 1: LOAD YOUR ASPECT-EXTRACTED DATA
# ============================================================================

class DataLoader:
    """Load the aspect-extracted dataset from Analysis 1"""
    
    def __init__(self, filepath='aspect_extraction_complete.csv'):
        self.filepath = filepath
        
    def load_data(self):
        """Load and prepare data"""
        print("="*70)
        print("ANALYSIS 2: ASPECT-BASED SENTIMENT ANALYSIS (ABSA)")
        print("="*70)
        print("\nSTEP 1: Loading Aspect-Extracted Data")
        print("-"*70)
        
        df = pd.read_csv(self.filepath)
        
        # Convert string lists back to lists if needed
        if isinstance(df['benefits_found'].iloc[0], str):
            import ast
            df['benefits_found'] = df['benefits_found'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )
            df['side_effects_found'] = df['side_effects_found'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )
        
        print(f"✓ Loaded {len(df)} posts with aspect annotations")
        print(f"  - Posts with benefits: {(df['benefits_count'] > 0).sum()}")
        print(f"  - Posts with side effects: {(df['side_effect_count'] > 0).sum()}")
        print(f"  - Posts with both: {((df['benefits_count'] > 0) & (df['side_effect_count'] > 0)).sum()}")
        
        return df

# ============================================================================
# STEP 2: IMPLEMENT SENTIMENT ANALYZERS
# ============================================================================

class SentimentAnalyzers:
    """Multiple sentiment analysis methods for triangulation"""
    
    def __init__(self):
        global TRANSFORMER_AVAILABLE  # Use global variable
        
        # VADER - Best for social media
        self.vader = SentimentIntensityAnalyzer()
        
        # TextBlob - Good for subjectivity
        # (imported directly)
        
        # Transformer - Most accurate (if available)
        self.transformer = None
        if TRANSFORMER_AVAILABLE:
            try:
                self.transformer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    truncation=True,
                    max_length=512
                )
                print("✓ Loaded transformer model (RoBERTa)")
            except Exception as e:
                print(f"⚠ Could not load transformer: {e}")
                TRANSFORMER_AVAILABLE = False
    
    def analyze_vader(self, text):
        """
        VADER Sentiment Analysis
        
        Method: Lexicon + rules-based
        Strengths: Great for social media, handles slang
        Output: compound score (-1 to +1)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        scores = self.vader.polarity_scores(text)
        return scores['compound']  # -1 (most negative) to +1 (most positive)
    
    def analyze_textblob(self, text):
        """
        TextBlob Sentiment Analysis
        
        Method: Pattern-based
        Strengths: Simplicity, subjectivity detection
        Output: polarity (-1 to +1)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_transformer(self, text):
        """
        Transformer-based Sentiment (RoBERTa)
        
        Method: Deep learning
        Strengths: Context understanding, highest accuracy
        Output: score converted to -1 to +1 scale
        """
        if not self.transformer or not text or len(text.strip()) == 0:
            return None
        
        try:
            result = self.transformer(text[:512])[0]  # Truncate if too long
            
            # Convert to -1 to +1 scale
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                return score
            elif 'negative' in label:
                return -score
            else:  # neutral
                return 0.0
        except:
            return None
    
    def analyze_all(self, text):
        """Run all available analyzers"""
        results = {
            'vader': self.analyze_vader(text),
            'textblob': self.analyze_textblob(text)
        }
        
        if self.transformer:
            transformer_score = self.analyze_transformer(text)
            if transformer_score is not None:
                results['transformer'] = transformer_score
        
        # Calculate ensemble (average)
        scores = [v for v in results.values() if v is not None]
        results['ensemble'] = np.mean(scores) if scores else 0.0
        
        return results

# ============================================================================
# STEP 3: EXTRACT ASPECT-SPECIFIC SENTIMENT
# ============================================================================

class AspectSentimentExtractor:
    """Extract sentiment for each aspect mentioned in text"""
    
    def __init__(self):
        self.analyzers = SentimentAnalyzers()
        
        # Load aspect keywords from Analysis 1
        self.aspect_keywords = self._load_aspect_keywords()
    
    def _load_aspect_keywords(self):
        """Define aspect keywords (from Analysis 1)"""
        return {
            # Benefits
            'weight_loss': ['weight', 'pounds', 'lbs', 'kg', 'lost', 'loss', 'lighter', 'slim'],
            'glucose_control': ['glucose', 'blood sugar', 'a1c', 'diabetes', 'insulin'],
            'appetite_suppression': ['appetite', 'hunger', 'cravings', 'food noise'],
            'health_improvement': ['health', 'healthy', 'energy', 'active', 'fitness'],
            'confidence': ['confidence', 'self-esteem', 'happy', 'proud'],
            
            # Side Effects
            'nausea': ['nausea', 'nauseous', 'sick', 'queasy'],
            'vomiting': ['vomit', 'vomiting', 'throw up', 'puke'],
            'diarrhea': ['diarrhea', 'loose stool', 'bowel'],
            'constipation': ['constipation', 'constipated', 'blocked'],
            'fatigue': ['fatigue', 'tired', 'exhausted', 'weak', 'weakness'],
            'headache': ['headache', 'migraine', 'head pain'],
            'stomach_issues': ['stomach', 'abdominal', 'belly', 'cramping', 'bloating'],
            'reflux': ['reflux', 'heartburn', 'acid', 'gerd', 'burp'],
            'injection_site': ['injection site', 'needle', 'bruising']
        }
    
    def extract_aspect_sentences(self, text, aspect_name):
        """
        Extract sentences mentioning a specific aspect
        
        Method: Sentence-level aspect detection
        Why: Sentiment is more accurate at sentence level than document level
        """
        if not text:
            return []
        
        # Get keywords for this aspect
        keywords = self.aspect_keywords.get(aspect_name, [])
        
        # Split into sentences - using simple regex to avoid NLTK issues
        # This splits on . ! ? followed by space or end of string
        import re
        sentences = re.split(r'[.!?]+\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Find sentences containing aspect keywords
        aspect_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in keywords):
                aspect_sentences.append(sent)
        
        return aspect_sentences
    
    def analyze_aspect_sentiment(self, text, aspect_name):
        """
        Analyze sentiment toward a specific aspect
        
        Process:
        1. Find sentences mentioning the aspect
        2. Analyze sentiment of each sentence
        3. Average sentiments (if multiple mentions)
        """
        # Extract sentences about this aspect
        aspect_sentences = self.extract_aspect_sentences(text, aspect_name)
        
        if not aspect_sentences:
            return None  # Aspect not mentioned
        
        # Analyze sentiment of each sentence
        sentiments = []
        for sent in aspect_sentences:
            sent_scores = self.analyzers.analyze_all(sent)
            sentiments.append(sent_scores['ensemble'])
        
        # Return average sentiment (if multiple mentions)
        return {
            'mean_sentiment': np.mean(sentiments),
            'max_sentiment': np.max(sentiments),
            'min_sentiment': np.min(sentiments),
            'sentence_count': len(sentiments),
            'sentences': aspect_sentences
        }
    
    def analyze_all_aspects(self, text, aspects_found):
        """Analyze sentiment for all aspects mentioned in the text"""
        aspect_sentiments = {}
        
        for aspect in aspects_found:
            result = self.analyze_aspect_sentiment(text, aspect)
            if result:
                aspect_sentiments[aspect] = result['mean_sentiment']
        
        return aspect_sentiments

# ============================================================================
# STEP 4: PROCESS ENTIRE DATASET
# ============================================================================

class ABSAProcessor:
    """Process entire dataset for aspect-based sentiment"""
    
    def __init__(self, df):
        self.df = df
        self.extractor = AspectSentimentExtractor()
        
    def process_dataset(self):
        """Apply ABSA to all posts"""
        
        print("\n" + "="*70)
        print("STEP 2: Performing Aspect-Based Sentiment Analysis")
        print("="*70)
        print("\nThis may take 5-10 minutes for large datasets...")
        print("Progress updates every 100 posts\n")
        
        # Initialize result columns
        all_aspects = list(self.extractor.aspect_keywords.keys())
        
        for aspect in all_aspects:
            self.df[f'sentiment_{aspect}'] = None
        
        # Process each post
        total = len(self.df)
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"  Processed {idx}/{total} posts ({idx/total*100:.1f}%)")
            
            text = row['text']
            
            # Get aspects mentioned in this post
            benefits = row['benefits_found'] if isinstance(row['benefits_found'], list) else []
            side_effects = row['side_effects_found'] if isinstance(row['side_effects_found'], list) else []
            all_aspects_in_post = benefits + side_effects
            
            # Analyze sentiment for each aspect
            aspect_sentiments = self.extractor.analyze_all_aspects(text, all_aspects_in_post)
            
            # Store results
            for aspect, sentiment in aspect_sentiments.items():
                self.df.at[idx, f'sentiment_{aspect}'] = sentiment
        
        print(f"\n✓ Completed ABSA for {total} posts")
        
        return self.df
    
    def calculate_aspect_aggregates(self):
        """Calculate average sentiment per aspect across all posts"""
        
        print("\n" + "="*70)
        print("STEP 3: Calculating Aggregate Aspect Sentiments")
        print("="*70)
        
        aspect_stats = {}
        
        # Benefits
        benefit_aspects = ['weight_loss', 'glucose_control', 'appetite_suppression', 
                          'health_improvement', 'confidence']
        
        # Side Effects
        side_effect_aspects = ['nausea', 'vomiting', 'diarrhea', 'constipation',
                              'fatigue', 'headache', 'stomach_issues', 'reflux', 
                              'injection_site']
        
        print("\n" + "-"*70)
        print("BENEFIT SENTIMENTS (Expected: Positive)")
        print("-"*70)
        print(f"{'Aspect':<30} {'Mean':<10} {'Std':<10} {'N':<10}")
        print("-"*70)
        
        for aspect in benefit_aspects:
            col = f'sentiment_{aspect}'
            if col in self.df.columns:
                values = self.df[col].dropna()
                if len(values) > 0:
                    aspect_stats[aspect] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'count': len(values),
                        'type': 'benefit'
                    }
                    print(f"{aspect.replace('_', ' ').title():<30} "
                          f"{values.mean():>+.3f}    {values.std():.3f}    {len(values):<10}")
        
        print("\n" + "-"*70)
        print("SIDE EFFECT SENTIMENTS (Expected: Negative)")
        print("-"*70)
        print(f"{'Aspect':<30} {'Mean':<10} {'Std':<10} {'N':<10}")
        print("-"*70)
        
        for aspect in side_effect_aspects:
            col = f'sentiment_{aspect}'
            if col in self.df.columns:
                values = self.df[col].dropna()
                if len(values) > 0:
                    aspect_stats[aspect] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'count': len(values),
                        'type': 'side_effect'
                    }
                    print(f"{aspect.replace('_', ' ').title():<30} "
                          f"{values.mean():>+.3f}    {values.std():.3f}    {len(values):<10}")
        
        return aspect_stats

# ============================================================================
# STEP 5: STATISTICAL ANALYSIS
# ============================================================================

class ABSAStatistics:
    """Statistical tests for aspect sentiment comparisons"""
    
    def __init__(self, df, aspect_stats):
        self.df = df
        self.aspect_stats = aspect_stats
    
    def test_benefit_vs_side_effect(self):
        """
        Test: Are benefits more positive than side effects are negative?
        
        H0: mean(benefit_sentiment) = -mean(side_effect_sentiment)
        H1: mean(benefit_sentiment) > -mean(side_effect_sentiment)
        
        This tests cognitive reweighting!
        """
        print("\n" + "="*70)
        print("STEP 4: Statistical Hypothesis Testing")
        print("="*70)
        
        # Get all benefit sentiments
        benefit_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                       if self.aspect_stats[a]['type'] == 'benefit']
        benefit_values = []
        for col in benefit_cols:
            if col in self.df.columns:
                benefit_values.extend(self.df[col].dropna().tolist())
        
        # Get all side effect sentiments
        side_effect_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                           if self.aspect_stats[a]['type'] == 'side_effect']
        side_effect_values = []
        for col in side_effect_cols:
            if col in self.df.columns:
                side_effect_values.extend(self.df[col].dropna().tolist())
        
        # Statistical test
        print("\n[TEST 1] Benefits vs. Side Effects")
        print("-"*70)
        
        mean_benefit = np.mean(benefit_values)
        mean_side_effect = np.mean(side_effect_values)
        
        print(f"Mean Benefit Sentiment:      {mean_benefit:>+.3f}")
        print(f"Mean Side Effect Sentiment:  {mean_side_effect:>+.3f}")
        print(f"Absolute Gap:                {mean_benefit - mean_side_effect:>+.3f}")
        
        # T-test
        t_stat, p_value = stats.ttest_ind(benefit_values, side_effect_values)
        
        print(f"\nIndependent t-test:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.001:
            print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.05:
            print(f"  Result: SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(benefit_values)**2 + np.std(side_effect_values)**2) / 2)
        cohens_d = (mean_benefit - mean_side_effect) / pooled_std
        
        print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
        if abs(cohens_d) > 0.8:
            print("  Interpretation: LARGE effect")
        elif abs(cohens_d) > 0.5:
            print("  Interpretation: MEDIUM effect")
        else:
            print("  Interpretation: SMALL effect")
        
        return {
            'mean_benefit': mean_benefit,
            'mean_side_effect': mean_side_effect,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d
        }
    
    def test_asymmetry_hypothesis(self):
        """
        Test: Is |benefit_sentiment| > |side_effect_sentiment|?
        
        This tests if benefits are MORE positive than side effects are negative
        = Evidence of cognitive minimization of harms
        """
        print("\n[TEST 2] Asymmetry Hypothesis (Cognitive Reweighting)")
        print("-"*70)
        
        # Calculate absolute values
        benefit_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                       if self.aspect_stats[a]['type'] == 'benefit']
        side_effect_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                           if self.aspect_stats[a]['type'] == 'side_effect']
        
        abs_benefits = []
        for col in benefit_cols:
            if col in self.df.columns:
                abs_benefits.extend(self.df[col].dropna().abs().tolist())
        
        abs_side_effects = []
        for col in side_effect_cols:
            if col in self.df.columns:
                abs_side_effects.extend(self.df[col].dropna().abs().tolist())
        
        mean_abs_benefit = np.mean(abs_benefits)
        mean_abs_side_effect = np.mean(abs_side_effects)
        
        print(f"Mean |Benefit Sentiment|:      {mean_abs_benefit:.3f}")
        print(f"Mean |Side Effect Sentiment|:  {mean_abs_side_effect:.3f}")
        print(f"Difference:                    {mean_abs_benefit - mean_abs_side_effect:+.3f}")
        
        if mean_abs_benefit > mean_abs_side_effect:
            print("\n✓ ASYMMETRY DETECTED: Benefits are MORE positive than side effects are negative")
            print("  → Evidence of cognitive reweighting (benefits amplified, harms minimized)")
        else:
            print("\n✗ No asymmetry: Symmetric evaluation")

# ============================================================================
# STEP 6: EXPLICIT WITHIN-POST TRADE-OFF ANALYSIS
# ============================================================================

class TradeOffAnalyzer:
    """
    Explicit Within-Post Trade-Off Analysis
    
    Tests compensatory decision-making by comparing benefit vs side-effect
    sentiment within the SAME post.
    """

    def __init__(self, df, aspect_stats):
        self.df = df
        self.aspect_stats = aspect_stats

    def run_tradeoff_analysis(self):

        print("\n" + "="*70)
        print("STEP 5: EXPLICIT WITHIN-POST TRADE-OFF ANALYSIS")
        print("="*70)

        # Identify benefit and side-effect columns dynamically
        benefit_cols = [
            f'sentiment_{a}' for a in self.aspect_stats.keys()
            if self.aspect_stats[a]['type'] == 'benefit'
        ]

        side_effect_cols = [
            f'sentiment_{a}' for a in self.aspect_stats.keys()
            if self.aspect_stats[a]['type'] == 'side_effect'
        ]

        # Select posts mentioning BOTH
        df_trade = self.df[
            (self.df['benefits_count'] > 0) &
            (self.df['side_effect_count'] > 0)
        ].copy()

        print(f"Posts mentioning BOTH benefits and side effects: {len(df_trade)}")

        if len(df_trade) == 0:
            print("No posts contain both aspects. Trade-off analysis not possible.")
            return None

        # Compute mean benefit and side-effect sentiment per post
        df_trade['mean_benefit_sentiment'] = df_trade[benefit_cols].mean(axis=1, skipna=True)
        df_trade['mean_side_effect_sentiment'] = df_trade[side_effect_cols].mean(axis=1, skipna=True)

        # Trade-off score
        df_trade['tradeoff_score'] = (
            df_trade['mean_benefit_sentiment'] -
            abs(df_trade['mean_side_effect_sentiment'])
        )

        # Summary statistics
        mean_benefit = df_trade['mean_benefit_sentiment'].mean()
        mean_side = df_trade['mean_side_effect_sentiment'].mean()
        mean_tradeoff = df_trade['tradeoff_score'].mean()

        print("\nWithin-Post Means:")
        print(f"  Mean Benefit Sentiment:      {mean_benefit:+.3f}")
        print(f"  Mean Side Effect Sentiment:  {mean_side:+.3f}")
        print(f"  Mean Trade-Off Score:        {mean_tradeoff:+.3f}")

        # Paired t-test (stronger than independent test here)
        t_stat, p_value = stats.ttest_rel(
            df_trade['mean_benefit_sentiment'],
            df_trade['mean_side_effect_sentiment']
        )

        print("\nPaired t-test (within-post comparison):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

        # Effect size (paired Cohen's d)
        diff = (
            df_trade['mean_benefit_sentiment'] -
            df_trade['mean_side_effect_sentiment']
        )
        cohens_d = diff.mean() / diff.std()

        print(f"\nEffect Size (Paired Cohen's d): {cohens_d:.3f}")

        if abs(cohens_d) > 0.8:
            print("  Interpretation: LARGE effect")
        elif abs(cohens_d) > 0.5:
            print("  Interpretation: MEDIUM effect")
        else:
            print("  Interpretation: SMALL effect")

        # Dominance rate
        dominance_rate = (df_trade['tradeoff_score'] > 0).mean()

        print(f"\nBenefit Dominance Rate: {dominance_rate*100:.2f}%")

        if mean_tradeoff > 0:
            print("\n✓ TRADE-OFF CONFIRMED")
            print("  Within the same post, benefits outweigh harms.")
            print("  → Strong evidence of compensatory decision-making.")
        else:
            print("\n✗ No systematic trade-off detected.")

        # Save trade-off dataset
        df_trade.to_csv('analysis2_within_post_tradeoff.csv', index=False)
        print("\n✓ Trade-off dataset saved as 'analysis2_within_post_tradeoff.csv'")

        return df_trade

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

class ABSAVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, df, aspect_stats):
        self.df = df
        self.aspect_stats = aspect_stats
    
    def create_dashboard(self):
        """Generate comprehensive visualization dashboard"""
        
        print("\n" + "="*70)
        print("STEP 6: Generating Visualizations")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Benefit Sentiments
        benefit_data = [(k.replace('_', ' ').title(), v['mean']) 
                       for k, v in self.aspect_stats.items() 
                       if v['type'] == 'benefit']
        benefit_data.sort(key=lambda x: x[1], reverse=True)
        
        aspects_b = [x[0] for x in benefit_data]
        values_b = [x[1] for x in benefit_data]
        
        axes[0, 0].barh(aspects_b, values_b, color='green', alpha=0.7)
        axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 0].set_xlabel('Mean Sentiment Score')
        axes[0, 0].set_title('Benefit Aspect Sentiments (Expected: Positive)', 
                            fontweight='bold', fontsize=12)
        axes[0, 0].set_xlim(-1, 1)
        
        # Plot 2: Side Effect Sentiments
        se_data = [(k.replace('_', ' ').title(), v['mean']) 
                   for k, v in self.aspect_stats.items() 
                   if v['type'] == 'side_effect']
        se_data.sort(key=lambda x: x[1])
        
        aspects_se = [x[0] for x in se_data]
        values_se = [x[1] for x in se_data]
        
        axes[0, 1].barh(aspects_se, values_se, color='red', alpha=0.7)
        axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].set_xlabel('Mean Sentiment Score')
        axes[0, 1].set_title('Side Effect Aspect Sentiments (Expected: Negative)', 
                            fontweight='bold', fontsize=12)
        axes[0, 1].set_xlim(-1, 1)
        
        # Plot 3: Comparison (Benefits vs Side Effects)
        comparison_data = pd.DataFrame({
            'Category': ['Benefits', 'Side Effects'],
            'Mean Sentiment': [
                np.mean([v['mean'] for k, v in self.aspect_stats.items() if v['type'] == 'benefit']),
                np.mean([v['mean'] for k, v in self.aspect_stats.items() if v['type'] == 'side_effect'])
            ]
        })
        
        bars = axes[1, 0].bar(comparison_data['Category'], 
                             comparison_data['Mean Sentiment'],
                             color=['green', 'red'], alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 0].set_ylabel('Mean Sentiment Score')
        axes[1, 0].set_title('Overall: Benefits vs. Side Effects', 
                            fontweight='bold', fontsize=12)
        axes[1, 0].set_ylim(-1, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 4: Distribution Comparison
        benefit_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                       if self.aspect_stats[a]['type'] == 'benefit']
        side_effect_cols = [f'sentiment_{a}' for a in self.aspect_stats.keys() 
                           if self.aspect_stats[a]['type'] == 'side_effect']
        
        all_benefit_vals = []
        for col in benefit_cols:
            if col in self.df.columns:
                all_benefit_vals.extend(self.df[col].dropna().tolist())
        
        all_se_vals = []
        for col in side_effect_cols:
            if col in self.df.columns:
                all_se_vals.extend(self.df[col].dropna().tolist())
        
        axes[1, 1].hist(all_benefit_vals, bins=30, alpha=0.6, 
                       color='green', label='Benefits', density=True)
        axes[1, 1].hist(all_se_vals, bins=30, alpha=0.6, 
                       color='red', label='Side Effects', density=True)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Sentiment Distribution Comparison', 
                            fontweight='bold', fontsize=12)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('analysis2_absa_results.png', dpi=300, bbox_inches='tight')
        
        print("\n✓ Visualization saved as 'analysis2_absa_results.png'")

# ============================================================================
# STEP 7: MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete Analysis 2: ABSA"""
    
    # Step 1: Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Step 2: Process ABSA
    processor = ABSAProcessor(df)
    df = processor.process_dataset()
    aspect_stats = processor.calculate_aspect_aggregates()
    
    # Step 3: Statistical tests
    stats_analyzer = ABSAStatistics(df, aspect_stats)
    test_results = stats_analyzer.test_benefit_vs_side_effect()
    stats_analyzer.test_asymmetry_hypothesis()

    # Step 4: Explicit Within-Post Trade-Off Analysis
    tradeoff_analyzer = TradeOffAnalyzer(df, aspect_stats)
    df_tradeoff = tradeoff_analyzer.run_tradeoff_analysis()
    
    # Step 5: Visualizations
    visualizer = ABSAVisualizer(df, aspect_stats)
    visualizer.create_dashboard()
    
    # Step 6: Save results
    print("\n" + "="*70)
    print("STEP 7: Saving Results")
    print("="*70)
    
    df.to_csv('analysis2_absa_complete.csv', index=False)
    print("\n✓ Complete dataset saved to 'analysis2_absa_complete.csv'")
    
    # Save aspect statistics
    aspect_stats_df = pd.DataFrame(aspect_stats).T
    aspect_stats_df.to_csv('analysis2_aspect_statistics.csv')
    print("✓ Aspect statistics saved to 'analysis2_aspect_statistics.csv'")
    
    # Summary report
    print("\n" + "="*70)
    print("ANALYSIS 2 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. analysis2_absa_complete.csv - Full dataset with aspect sentiments")
    print("  2. analysis2_aspect_statistics.csv - Aggregate sentiment per aspect")
    print("  3. analysis2_absa_results.png - Visualization dashboard")
    
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)
    print(f"Mean Benefit Sentiment:      {test_results['mean_benefit']:>+.3f}")
    print(f"Mean Side Effect Sentiment:  {test_results['mean_side_effect']:>+.3f}")
    print(f"Statistical Significance:    p = {test_results['p_value']:.4f}")
    print(f"Effect Size (Cohen's d):     {test_results['cohens_d']:.3f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if test_results['mean_benefit'] > abs(test_results['mean_side_effect']):
        print("✓ ASYMMETRIC EVALUATION CONFIRMED")
        print("  Benefits are MORE positive than side effects are negative")
        print("  → Evidence of cognitive reweighting")
        print("  → Supports compensatory decision-making model")
    
    
    
    return df, aspect_stats, test_results

if __name__ == "__main__":
    df_results, aspect_statistics, statistical_tests = main()