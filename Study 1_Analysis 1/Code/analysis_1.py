#basic library for visualization
import pandas as pd
import numpy as np
import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

#NLP libraries
import spacy
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import nltk

#dowloading the reuired nltk data for corpus matching
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

#step -1 to load and prepare the data form the analysis

class DataLoader:
    def __init__(self, primary_path, secondary_path):
        self.primary_path = primary_path
        self.secondary_path = secondary_path
    
    def load_data(self):
        print("="*30)
        print("Step 1 : Loading the data to transform")
        print("="*30)

        #load primary data(reddit)
        df_primary = pd.read_csv(self.primary_path)
        df_primary['source'] = 'Reddit'
        df_primary['text'] = df_primary['post_text'].fillna('')
        print(f"Loaded {len(df_primary)} Reddit posts")

        #load of secondary data(review)
        df_secondary = pd.read_csv(self.secondary_path)
        df_secondary['source'] = 'Review Platform'
        df_secondary['text'] = df_secondary['review_text'].fillna('')
        print(f"Loaded {len(df_secondary)} Platform reviews")

        #combine datasets
        df = pd.concat([
            df_primary[['text','source']],
            df_secondary[['text','source']]
        ], ignore_index=True)

        #Remove empty texts
        df = df[df['text'].str.len() > 10]

        print(f"\n Combined Dataset: {len(df)} total posts")
        print(f"Reddit: {(df['source'] == 'Reddit').sum()} posts")
        print(f"Reviews: {(df['source'] == 'Review Platform').sum()} posts")

        return df
    
# #step -2 - keywords declaration Dictionary based approach (Deduction) 
class AspectDictionary:
    def __init__(self):
        self.benefits = {
            'weight_loss' : {
                'keywords' :[
                    'weight', 'pounds', 'lbs', 'kg', 'lost','loss','lighter',
                    'slim', 'skinny', 'dropped', 'shed', 'size', 'dress size',
                    'pants size'
                ],
                'description' : 'weight reduction and body composition changes'
            },
            'glucose_control': {
                'keywords' : [
                    'glucose','blood sugar', 'a1c', 'hemoglobin a1c', 'diabetes',
                    'diabetic', 'insulin','sugar level', 'glycemic', 'blood glucose'
                ],
                'description' : 'Blood suage management and diabetes control'
            },

            'appetite_suppression' : {
                'keywords' : [
                    'appetite', 'hunger', 'cravings', 'food noise','eat less', 'full', 
                    'satiety', 'satisfied', 'portion','overeating', 'binge'
                ],
                'description' : 'Reduced appetite and eating behavior changes'
            },
            
            'health_improvement' : {
                'keywords' : [
                    'health', 'healthy','healthier', 'energy', 'active', 'fitness',
                    'exercise','mobility','cholesterol','blood pressure','cardiovascular'
                ],
                'description' : 'General health and wellness improvements'
            },

            'confidence' : {
                'keywords' : [
                    'confidence','confident','self-esteem','happy',
                    'proud','better about myself', 'self-image',
                    'mental health','depression','anxiety'
                ],
                'description' : 'Psycholgical and emotional benefits'
            }
        }

#SIDE EFFECTS - Based on FDA documentation and user reports
        self.side_effects = {
            'nausea' : {
                'keywords' : [
                    'nausea','nauseous', 'nauseated', 'sick', 'queasy',
                    'sick to stomach', 'morning sickness'
                ],
                'description' : 'Feeling of needing to vomit'
            },

            'vomiting' : {
                'keywords' : [
                    'vomit','vomitting','throw up', 'throwing up',
                    'puking', 'puke', 'threw up'
                ],
                'description' : 'Actual vomitting episodes'
            },

            'diarrhea': {
                'keywords' : [
                    'diarrhea', 'losse stool', 'loose stools','bowel movement',
                    'bathroom', 'runs'
                ],
                'description' : "Loose or liquid bowel movements"
            },

            'constipation' : {
                'keywords' :[
                    'constipation', 'constipated', 'blocked','backed up', 'cant go',
                    "can't go", 'irregular'
                ],
                'description': 'Difficulty passing stools'
            },

            'fatigue' : {
                'keywords' : [
                    'fatigue', 'tred', 'exhausted', 'exhaustion','weak',
                    'weakness', 'energy', 'lethargic', 'sluggish'
                ],
                'description': 'Extreme tiredness and low energy'
            },

            'headache' : {
                'keywords' : [
                    'headache', 'head ache', 'migraine', 'head pain',
                    'head hurts', 'head pounding'
                ],
                'descrption': 'Head pain'
            },

            'stomach_issues' : {
                'keywords' : [
                    'stomach', 'abdominal', 'belly', 'gastric','stomach pain',
                    'stomach ache','cramping','bloating', 'gas', 'indesgition'
                ],
                'description' : 'Stomach and abdominal discomfort'
            },

            'reflux' : {
                'keywords' : [
                    'reflux', 'heartburn', 'acid', 'gerd','burping', 'burps','sulfur burp'
                ],
                'description': 'Acid reflux and heartburn'
            },

            'injection_site' : {
                'keywords' : [
                    'injection site', 'injection', 'needle', 'bruising','swelling','redness','itching at site'
                ],
                'description': 'Reaction at injection location'
            }
        }
    
    def get_all_aspects(self):
        return {
            'benefits' : self.benefits,
            'side_effects' : self.side_effects
        }
    
#Step -3 - Implement aspect extraction methods
class AspectExtractor:
    
    def __init__(self):
        self.aspect_dict = AspectDictionary()
        self.nlp = spacy.load('en_core_web_sm')
    
    #dictionary based extraction methods
    def extract_dictionary_based(self, text):
        
        text_lower = text.lower()

        result = {
            'benefits_found' : [],
            'side_effects_found' :[],
            'benefits_matches' : {},
            'side_effect_matches' : {}
        }
    
#Search for benefits
        for aspect_name, aspect_data in self.aspect_dict.benefits.items():
            matches = []
            for keyword in aspect_data['keywords']:

#use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    matches.append(keyword)
            
            if matches:
                result['benefits_found'].append(aspect_name)
                result['benefits_matches'][aspect_name] = matches

        # Search for side effects
        for aspect_name, aspect_data in self.aspect_dict.side_effects.items():
            matches = []
            for keyword in aspect_data['keywords']:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    matches.append(keyword)
            
            if matches:
                result['side_effects_found'].append(aspect_name)
                result['side_effect_matches'][aspect_name] = matches


        return result
    
    #dependency based method
    def extract_dependency_based(self,text):
        doc = self.nlp(text)
        opinion_pairs = []

        for token in doc:
            # search for adjective (opinion words)
            if token.pos_ == 'ADJ':
                #find what adjective modifies
                for child in token.children:
                    if child.dep_ in ['nsubj','nsubjpass']:
                        opinion_pairs.append({
                            'aspect': child.text,
                            'opinion' : token.text,
                            'content' : token.sent.text
                        })
                
                #to check if adjective is describing the subject
                if token.head.pos_ in ['NOUN','PROPN']:
                    opinion_pairs.append({
                        'aspect': token.head.text,
                        'opinion' : token.text,
                        'context': token.sent.text
                    })
        return opinion_pairs
    
    def process_dataset(self,df):
        #printing the both methods
        print("\n" + "=" *30)
        print("Step 2: Extracting Aspects")
        print("="*30)
        print("\nThis may take time for process..\n")

        #dictionary based process
        print("[Method 1] Dictionary based extraction")
        dict_results = df['text'].apply(self.extract_dictionary_based)

        df['benefits_found'] = dict_results.apply(lambda x: x['benefits_found'])
        df['side_effects_found'] = dict_results.apply(lambda x: x['side_effects_found'])
        df['benefits_count'] = df['benefits_found'].apply(len)
        df['side_effect_count'] = df['side_effects_found'].apply(len)

        #Create a binary flags or hard encode for each aspect
        for aspect_name in self.aspect_dict.benefits.keys():
            df[f'has_benefit_{aspect_name}'] = df['benefits_found'].apply(
                lambda x: aspect_name in x
            )
        
        for aspect_name in self.aspect_dict.side_effects.keys():
            df[f'has_side_effect_{aspect_name}'] =df['side_effects_found'].apply(
                lambda x:aspect_name in x
            )

        print("Dictionary-based extraction complete")

        #Dependency parsing process
        print("\n[Method 2] Dpendency parsing extraction (sample)...")
        sample_size = min(500,len(df))
        sample_df = df.sample(n=sample_size,random_state=42)
        dependency_results = sample_df['text'].apply(self.extract_dependency_based)
        print(f"Dependency parsing complete (n={sample_size})")
        return df, dependency_results
    
#Step 4 Results

class AspectAnalyzer:
    def __init__(self,df,aspect_dict):
        self.df = df
        self.aspect_dict = aspect_dict
    
    def generate_summary_statistics(self):
        print("="*30)
        print("Step 3 : Summary")
        print("="*30)
        stats = {}

        stats['total_posts'] =len(self.df)
        stats['posts_with_benefits'] = (self.df['benefits_count'] > 0).sum()
        stats['posts_with_side_effects'] = (self.df['side_effect_count'] > 0).sum()
        stats['posts_with_both'] = ((self.df['benefits_count'] > 0 ) & 
                                    (self.df['side_effect_count'] > 0)).sum()
        stats['posts_with_neither'] = ((self.df['benefits_count'] == 0 ) & 
                                    (self.df['side_effect_count'] == 0)).sum()
        
        #Percentage calculation

        total = stats['total_posts']
        print(f"\n{'Metric':<40} {'Count':<10} {'%':<10}")
        print("-"*30)
        print(f"{'Total posts analyzed':<40} {total:100} {100:.1f}%")
        print(f"{'Posts mentioning benefits':<40} {stats['posts_with_benefits']:<10}"
              f"{stats['posts_with_benefits']/total*100:.1f}%")
        print(f"{'Posts mentioning side effects':<40} {stats['posts_with_side_effects']:<10}"
              f"{stats['posts_with_side_effects']/total*100:.1f}%")
        print(f"{'Posts mentioning BOTH':<40} {stats['posts_with_both']:<10}"
              f"{stats['posts_with_both']/total*100:.1f}%")
        print(f"{'Posts mentioning NEITHER':<40} {stats['posts_with_neither']:<10}"
              f"{stats['posts_with_neither']/total*100:.1f}%")
        
        #Aspect Frequency Analysis
        print("\n"+"="*30)
        print("Benefit Aspects - Frequency Ranking")
        print("="*30)

        benefit_freq = {}
        for aspect in self.aspect_dict.benefits.keys():
            count = self.df[f'has_benefit_{aspect}'].sum()
            benefit_freq[aspect] = count
        
        benefit_freq_sorted = sorted(benefit_freq.items(), key=lambda x:x[1],reverse=True)
        print(f"\n{'Benefit Aspect':<30} {'Count':<10} {'% of Posts':<15}")
        print("-"*30)
        for aspect, count in benefit_freq_sorted:
            print(f"{aspect.replace('_',' ').title():<30} {count:<10}"
                  f"{count/total*100:.1f}%")
            
        print("\n" + "="*30)
        print("Side Effect Aspects - Frequency Ranking")
        print("="*30)

        side_effect_freq = {}
        for aspect in self.aspect_dict.side_effects.keys():
            count = self.df[f'has_side_effect_{aspect}'].sum()
            side_effect_freq[aspect] = count
        
        side_effect_freq_sorted = sorted(side_effect_freq.items(),key=lambda x: x[1],reverse=True)
        print(f"\n{'Side Effect Aspect':<30} {'Count':<10} {'% of Posts':<15}")
        print("-"*30)
        for aspect, count in side_effect_freq_sorted:
            print(f"{aspect.replace('_',' ').title():<30} {count:<10}" f"{count/total*100:.1f}%")

        return stats, benefit_freq_sorted,side_effect_freq_sorted
    
    def create_visualizations(self, benefit_freq,side_effect_freq):
        print("\n"+"="*30)
        print("Generating Visualization")
        print("="*30)
        fig,axes = plt.subplots(2,2,figsize=(16,12))

        #To plot aspect (Benefits vs side effecs)
        aspect_comparison = pd.DataFrame({
            'Category':['Benefits','Side Effects'],
            'Count':[
                (self.df['benefits_count']>0).sum(),
                (self.df['side_effect_count']>0).sum()
            ]
        })

        sns.barplot(data=aspect_comparison, x='Category', y='Count', palette=['green', 'red'], ax=axes[0, 0])
        axes[0,0].set_title('Posts Mentioning Benefits vs Side Effects',fontsize=14,fontweight='bold')
        axes[0,0].set_ylabel('Number of posts')

        #Display of top 5 benefits
        top_benefits = benefit_freq[:5]
        benefit_names = [b[0].replace('_',' ').title() for b in top_benefits]
        benefit_counts = [b[1] for b in top_benefits]

        sns.barplot(x=benefit_counts, y=benefit_names,palette='Greens_r',ax=axes[0,1])
        axes[0,1].set_title('Top 5 Most Mentioned Benefits',fontsize=14,fontweight='bold')
        axes[0,1].set_xlabel('Number of Posts')

        #Display of Top 5 Side Effects
        top_side_effects = side_effect_freq[:5]
        se_names = [se[0].replace('_', ' ').title() for se in top_side_effects]
        se_count = [se[1] for se in top_side_effects]

        sns.barplot(x=se_count, y=se_names, palette='Reds_r', ax=axes[1,0])
        axes[1,0].set_title('Top 5 Most Metioned Side Effects',fontsize=14,fontweight='bold')
        axes[1,0].set_xlabel('Number of Posts')

        #To display the co-occurence of the word(bith vs either)
        co_occurence = pd.DataFrame({
            'Type':['Benefits only', 'Side Effects Only', 'Both', 'Neither'],
            'Count': [
                ((self.df['benefits_count'] > 0) & (self.df['side_effect_count'] == 0)).sum(),
                 ((self.df['benefits_count'] == 0) & (self.df['side_effect_count'] > 0)).sum(),
                  ((self.df['benefits_count'] > 0) & (self.df['side_effect_count'] > 0)).sum(),
                   ((self.df['benefits_count'] == 0) & (self.df['side_effect_count'] == 0)).sum()
            ]
        })

        axes[1,1].pie(co_occurence['Count'], labels=co_occurence['Type'],autopct = '%1.1f%%', startangle=90,
                      colors=['lightgreen','lightcoral','gold','lightgray'])
        axes[1,1].set_title('Aspect co-occurence Patterns',fontsize=14,fontweight='bold')

        plt.tight_layout()

        plt.savefig('Aspect_Extraction_Result.png',dpi=300,bbox_inches='tight')
        print("\n Visulization is saved")

    def extract_example_posts(self):
        print("\n"+"="*30)
        print("Extracting the post info")
        print("="*30)

        example = []

        benefits_only = self.df[
            (self.df['benefits_count'] > 0) &
            (self.df['side_effect_count'] == 0)
        ].sample(min(5,len(self.df)), random_state=42)

        side_effects_only = self.df[
            (self.df['benefits_count'] == 0) &
            (self.df['side_effect_count'] > 0)
        ].sample(min(5,len(self.df)), random_state=42)

        both = self.df[
            (self.df['benefits_count'] > 0) &
            (self.df['side_effect_count'] > 0)
        ].sample(min(5,len(self.df)), random_state=42)

        example_df = pd.concat([
            benefits_only[['text','benefits_found','side_effects_found']],
            side_effects_only[['text','benefits_found','side_effects_found']],
            both[['text','benefits_found','side_effects_found']]
        ])

        example_df.to_csv('analysis1_aspect_posts.csv',index=False)

        print("\n Posts saved to csv file")
        print(f" -Benefits only : 5 examples")
        print(f" -Side effects only : 5 examples")
        print(f" -Both mentioned : 5 examples")

        #Main Execution
def main():
        print("\n"+"="*30)
        print("Analysis 1: Aspect Extraction")
        print("Problem Statment 1:Cogonitive Trade-offs Between Benefits and Side Effects")
        print("="*30)

        #code line to load the data
        loader = DataLoader(
            primary_path='F:/MBA 2025-27/SEM 2/3_CREDIT/Research structure/Study 1/Study 1_Analysis 1/Dataset/primary_ozempic_reddit_data.csv',
            secondary_path='F:/MBA 2025-27/SEM 2/3_CREDIT/Research structure/Study 1/Study 1_Analysis 1/Dataset/secondary_ozempic_reviews_data.csv'
        )
        df=loader.load_data()

        #code line to execute the extarct aspect method
        extractor = AspectExtractor()
        df, dependency_results = extractor.process_dataset(df)

        #code line to analyze the results
        analyzer = AspectAnalyzer(df,extractor.aspect_dict)
        stats, benefit_freq, side_effect_freq = analyzer.generate_summary_statistics()
        analyzer.create_visualizations(benefit_freq,side_effect_freq)
        analyzer.extract_example_posts()

        #to save the complete results
        print("\n"+ "="*30)
        print("Saving the Final results")
        print("="*30)
        df.to_csv('aspect_extraction_complete.csv',index=False)
        print("\n Complete dataset saved as csv file")

        print("\n"+"="*30)
        print("Analysis 1 - Aspect Extraction Complete")
        print("="*30)
        
        return df,stats

if __name__ == "__main__":
    df_results,statistics = main()

