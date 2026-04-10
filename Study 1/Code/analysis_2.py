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
import re

# Try to load transformer model (optional)
TRANSFORMER_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMER_AVAILABLE = True
    print("✓ Transformer models available")
except:
    TRANSFORMER_AVAILABLE = False
    print("⚠ Transformer models not available (using VADER + TextBlob)")

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
        print(f"  - Posts with BOTH (trade-off candidates): {((df['benefits_count'] > 0) & (df['side_effect_count'] > 0)).sum()}")
        
        return df

# ============================================================================
# STEP 2: IMPLEMENT SENTIMENT ANALYZERS
# ============================================================================

class SentimentAnalyzers:
    """Multiple sentiment analysis methods for triangulation"""
    
    def __init__(self):
        # VADER - Best for social media
        self.vader = SentimentIntensityAnalyzer()
        
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
    
    def analyze_vader(self, text):
        """VADER Sentiment Analysis"""
        if not text or len(text.strip()) == 0:
            return 0.0
        scores = self.vader.polarity_scores(text)
        return scores['compound']
    
    def analyze_textblob(self, text):
        """TextBlob Sentiment Analysis"""
        if not text or len(text.strip()) == 0:
            return 0.0
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_transformer(self, text):
        """Transformer-based Sentiment (RoBERTa)"""
        if not self.transformer or not text or len(text.strip()) == 0:
            return None
        try:
            result = self.transformer(text[:512])[0]
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                return score
            elif 'negative' in label:
                return -score
            else:
                return 0.0
        except:
            return None
    
    def analyze_all(self, text):
        """Run all available analyzers and return ensemble"""
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
        """Extract sentences mentioning a specific aspect"""
        if not text:
            return []
        
        keywords = self.aspect_keywords.get(aspect_name, [])
        
        # Split into sentences using simple regex
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
        """Analyze sentiment toward a specific aspect"""
        aspect_sentences = self.extract_aspect_sentences(text, aspect_name)
        
        if not aspect_sentences:
            return None
        
        # Analyze sentiment of each sentence
        sentiments = []
        for sent in aspect_sentences:
            sent_scores = self.analyzers.analyze_all(sent)
            sentiments.append(sent_scores['ensemble'])
        
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
    
    def calculate_aggregate_sentiments(self):
        """
        Calculate aggregate sentiments per aspect
        This prepares summary statistics for trade-off analysis
        """
        
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
    
    def prepare_tradeoff_features(self):
        """
        NEW: Calculate trade-off indicators for Analysis 6
        
        This creates aggregate features that will be used in trade-off analysis
        """
        
        print("\n" + "="*70)
        print("STEP 4: Preparing Trade-Off Features (for Analysis 6)")
        print("="*70)
        
        benefit_cols = [col for col in self.df.columns if col.startswith('sentiment_') 
                       and any(b in col for b in ['weight', 'glucose', 'appetite', 'health', 'confidence'])]
        
        side_effect_cols = [col for col in self.df.columns if col.startswith('sentiment_') 
                           and any(se in col for se in ['nausea', 'vomit', 'diarrhea', 'constipation', 
                                                         'fatigue', 'headache', 'stomach', 'reflux', 'injection'])]
        
        # Calculate mean benefit and side effect sentiment per post
        self.df['mean_benefit_sentiment'] = self.df[benefit_cols].mean(axis=1, skipna=True)
        self.df['mean_side_effect_sentiment'] = self.df[side_effect_cols].mean(axis=1, skipna=True)
        
        # Calculate sentiment gap (magnitude difference)
        self.df['sentiment_gap'] = (
            self.df['mean_benefit_sentiment'] - 
            abs(self.df['mean_side_effect_sentiment'])
        )
        
        # Flag posts with both aspects (trade-off candidates)
        self.df['has_both_aspects'] = (
            (self.df['benefits_count'] > 0) & 
            (self.df['side_effect_count'] > 0)
        )
        
        # Calculate asymmetry (for each post)
        self.df['sentiment_asymmetry'] = (
            abs(self.df['mean_benefit_sentiment']) / 
            (abs(self.df['mean_side_effect_sentiment']) + 0.01)  # Avoid division by zero
        )
        
        print("\n✓ Trade-off features created:")
        print(f"  - mean_benefit_sentiment: Average sentiment across all benefits")
        print(f"  - mean_side_effect_sentiment: Average sentiment across all side effects")
        print(f"  - sentiment_gap: Benefit sentiment - |Side effect sentiment|")
        print(f"  - has_both_aspects: Boolean flag for trade-off candidates")
        print(f"  - sentiment_asymmetry: Ratio of |benefit| / |side effect|")
        
        # Summary statistics for trade-off candidates
        tradeoff_posts = self.df[self.df['has_both_aspects'] == True]
        
        if len(tradeoff_posts) > 0:
            print(f"\n✓ Trade-off candidates: {len(tradeoff_posts)} posts")
            print(f"  - Mean benefit sentiment: {tradeoff_posts['mean_benefit_sentiment'].mean():+.3f}")
            print(f"  - Mean side effect sentiment: {tradeoff_posts['mean_side_effect_sentiment'].mean():+.3f}")
            print(f"  - Mean sentiment gap: {tradeoff_posts['sentiment_gap'].mean():+.3f}")
            print(f"  - Posts where benefits > side effects: {(tradeoff_posts['sentiment_gap'] > 0).sum()} ({(tradeoff_posts['sentiment_gap'] > 0).mean()*100:.1f}%)")

# ============================================================================
# STEP 5: STATISTICAL ANALYSIS
# ============================================================================

class ABSAStatistics:
    """Statistical tests for aspect sentiment comparisons"""
    
    def __init__(self, df, aspect_stats):
        self.df = df
        self.aspect_stats = aspect_stats
    
    def test_benefit_vs_side_effect(self):
        """Test: Are benefits more positive than side effects are negative?"""
        
        print("\n" + "="*70)
        print("STEP 5: Statistical Hypothesis Testing")
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
        print("\n[TEST 1] Benefits vs. Side Effects (Cross-Corpus)")
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
    
    def test_within_post_comparison(self):
        """
        Test: Within posts mentioning BOTH, do benefits outweigh side effects?
        This is a PREVIEW of Analysis 6 (full trade-off analysis)
        """
        
        print("\n[TEST 2] Within-Post Comparison (Trade-Off Preview)")
        print("-"*70)
        
        # Filter to posts with both aspects
        tradeoff_posts = self.df[self.df['has_both_aspects'] == True].copy()
        
        if len(tradeoff_posts) < 10:
            print("⚠ Too few trade-off posts for reliable statistics")
            return None
        
        print(f"Posts with BOTH aspects: {len(tradeoff_posts)}")
        
        # Paired t-test (within-post comparison)
        valid_posts = tradeoff_posts[
            tradeoff_posts['mean_benefit_sentiment'].notna() & 
            tradeoff_posts['mean_side_effect_sentiment'].notna()
        ]
        
        if len(valid_posts) < 10:
            print("⚠ Too few valid pairs for paired t-test")
            return None
        
        t_stat, p_value = stats.ttest_rel(
            valid_posts['mean_benefit_sentiment'],
            valid_posts['mean_side_effect_sentiment']
        )
        
        print(f"\nWithin-post means:")
        print(f"  Benefit sentiment:      {valid_posts['mean_benefit_sentiment'].mean():+.3f}")
        print(f"  Side effect sentiment:  {valid_posts['mean_side_effect_sentiment'].mean():+.3f}")
        print(f"  Sentiment gap:          {valid_posts['sentiment_gap'].mean():+.3f}")
        
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.001:
            print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.05:
            print(f"  Result: SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        
        # Benefit dominance rate
        dominance = (valid_posts['sentiment_gap'] > 0).mean()
        print(f"\nBenefit Dominance Rate: {dominance*100:.1f}%")
        print(f"  (Percentage of posts where benefits outweigh side effects)")
        
        if dominance > 0.6:
            print("\n✓ Strong evidence of benefit dominance")
            print("  → Supports compensatory decision-making hypothesis")
        
        return {
            'dominance_rate': dominance,
            't_statistic': t_stat,
            'p_value': p_value
        }

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
        
        if benefit_data:
            aspects_b = [x[0] for x in benefit_data]
            values_b = [x[1] for x in benefit_data]
            
            axes[0, 0].barh(aspects_b, values_b, color='green', alpha=0.7)
            axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[0, 0].set_xlabel('Mean Sentiment Score')
            axes[0, 0].set_title('Benefit Aspect Sentiments', 
                                fontweight='bold', fontsize=12)
            axes[0, 0].set_xlim(-1, 1)
        
        # Plot 2: Side Effect Sentiments
        se_data = [(k.replace('_', ' ').title(), v['mean']) 
                   for k, v in self.aspect_stats.items() 
                   if v['type'] == 'side_effect']
        se_data.sort(key=lambda x: x[1])
        
        if se_data:
            aspects_se = [x[0] for x in se_data]
            values_se = [x[1] for x in se_data]
            
            axes[0, 1].barh(aspects_se, values_se, color='red', alpha=0.7)
            axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[0, 1].set_xlabel('Mean Sentiment Score')
            axes[0, 1].set_title('Side Effect Aspect Sentiments', 
                                fontweight='bold', fontsize=12)
            axes[0, 1].set_xlim(-1, 1)
        
        # Plot 3: Comparison
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
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom' if height > 0 else 'top')
        
        # Plot 4: Trade-off Distribution
        tradeoff_posts = self.df[self.df['has_both_aspects'] == True]
        
        if len(tradeoff_posts) > 0:
            axes[1, 1].hist(tradeoff_posts['sentiment_gap'], bins=30, 
                           color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, 
                              label='Neutral (no trade-off)')
            axes[1, 1].set_xlabel('Sentiment Gap (Benefit - |Side Effect|)')
            axes[1, 1].set_ylabel('Number of Posts')
            axes[1, 1].set_title('Trade-Off Distribution (Posts with Both Aspects)', 
                                fontweight='bold', fontsize=12)
            axes[1, 1].legend()
            
            # Add text showing dominance rate
            dominance = (tradeoff_posts['sentiment_gap'] > 0).mean()
            axes[1, 1].text(0.95, 0.95, f'Benefit Dominance:\n{dominance*100:.1f}%',
                           transform=axes[1, 1].transAxes,
                           fontsize=11, fontweight='bold',
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
    aspect_stats = processor.calculate_aggregate_sentiments()
    processor.prepare_tradeoff_features()  # NEW: Prepare for Analysis 6
    
    # Step 3: Statistical tests
    stats_analyzer = ABSAStatistics(df, aspect_stats)
    test_results = stats_analyzer.test_benefit_vs_side_effect()
    within_post_results = stats_analyzer.test_within_post_comparison()
    
    # Step 4: Visualizations
    visualizer = ABSAVisualizer(df, aspect_stats)
    visualizer.create_dashboard()
    
    # Step 5: Save results
    print("\n" + "="*70)
    print("STEP 7: Saving Results")
    print("="*70)
    
    df.to_csv('analysis2_absa_complete.csv', index=False)
    print("\n✓ Complete dataset saved to 'analysis2_absa_complete.csv'")
    print("  Contains: Aspect sentiments + trade-off features for Analysis 6")
    
    # Save aspect statistics
    aspect_stats_df = pd.DataFrame(aspect_stats).T
    aspect_stats_df.to_csv('analysis2_aspect_statistics.csv')
    print("✓ Aspect statistics saved to 'analysis2_aspect_statistics.csv'")
    
    # Save trade-off candidates for Analysis 6
    tradeoff_df = df[df['has_both_aspects'] == True].copy()
    if len(tradeoff_df) > 0:
        tradeoff_df.to_csv('analysis2_tradeoff_candidates.csv', index=False)
        print(f"✓ Trade-off candidates saved to 'analysis2_tradeoff_candidates.csv' ({len(tradeoff_df)} posts)")
    
    # Summary report
    print("\n" + "="*70)
    print("ANALYSIS 2 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. analysis2_absa_complete.csv - Full dataset with aspect sentiments")
    print("  2. analysis2_aspect_statistics.csv - Aggregate sentiment per aspect")
    print("  3. analysis2_tradeoff_candidates.csv - Posts for Analysis 6")
    print("  4. analysis2_absa_results.png - Visualization dashboard")
    
    print("\n" + "="*70)
    print("KEY FINDINGS SUMMARY")
    print("="*70)
    print(f"Mean Benefit Sentiment:      {test_results['mean_benefit']:>+.3f}")
    print(f"Mean Side Effect Sentiment:  {test_results['mean_side_effect']:>+.3f}")
    print(f"Statistical Significance:    p = {test_results['p_value']:.4f}")
    print(f"Effect Size (Cohen's d):     {test_results['cohens_d']:.3f}")
    
    if within_post_results:
        print(f"\nWithin-Post Analysis:")
        print(f"Benefit Dominance Rate:      {within_post_results['dominance_rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print("INTERPRETATION FOR YOUR RESEARCH")
    print("="*70)
    
    if test_results['mean_benefit'] > abs(test_results['mean_side_effect']):
        print("✓ ASYMMETRIC EVALUATION CONFIRMED")
        print("  Benefits are MORE positive than side effects are negative")
        print("  → Evidence of cognitive reweighting")
        print("  → Prepares ground for Analysis 6 (Trade-off Language Detection)")
    
     
    return df, aspect_stats, test_results

if __name__ == "__main__":
    df_results, aspect_statistics, statistical_tests = main()