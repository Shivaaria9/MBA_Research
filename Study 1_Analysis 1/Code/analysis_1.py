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
nltk.download('averaged_perception_tagger')

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
                'keyboard' : [
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
                'keyword' : [
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
                result['benefit_matches'][aspect_name] = matches

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
                    if child.dep_in ['nsubj','nsubjpass']:
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
        dict_results = df['text'].apply(self.extract_dependency_based)

        df['benefits_found'] = dict_results.apply(lambda x: x['benefits_found'])
        df['side_effects_found'] = dict_results.apply(lambda x: x['side_effects_found'])
        df['benefits_count'] = df['benefits_found'].apply(len)
        df['side_effect_count'] = df['side_effects_found'].apply(len)

        #Create a binary flags or hard encode for eacdh aspect
        for aspect_name in self.aspect_name.benefits.keys():
            df[f'has_benefit_{aspect_name}'] = df['benefit_found'].apply(
                lambda x: aspect_name in x
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
        stats['posts_with_benefits'] = (self.df['benefit_count'] > 0).sum()
        stats['posts_with_side_effects'] = (self.df['side_effects_count'] > 0).sum()
        stats['posts_with_both'] = ((self.df['benefit_count'] > 0 ) & 
                                    (self.df['side_effect_count'] > 0)).sum()
        stats['posts_with_neither'] = ((self.df['benefit_count'] == 0 ) & 
                                    (self.df['side_effect_count'] == 0)).sum()
        
        #Percentage calculation

        total = stats['total_posts']
        print(f"\n{'Metric':<40} {'Count':<10} {'%':<10}")
        print("-"*30)
        