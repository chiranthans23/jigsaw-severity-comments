import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
import re 
import scipy
from scipy import sparse
import gc 
from sklearn.model_selection import train_test_split,KFold
from pprint import pprint
import warnings
import nltk
import string
from gensim.models import KeyedVectors, FastText
import emoji
from collections import Counter
from spacy.lang.en import English
    
nltk.download('stopwords')
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")


pd.options.display.max_colwidth=300


from scipy.sparse import hstack

def splitter(text):
    tokens = []
    
    for word in text.split(' '):
        tokens.append(word)
    
    return tokens

def vectorizer(text,vec,fmodel):
    tokens = splitter(text)
    
    x1 = vec.transform([text]).toarray()
    x2 = np.mean(fmodel.wv[tokens], axis = 0).reshape(1, -1)
    x = np.concatenate([x1, x2], axis = -1).astype(np.float16)
    
    return x   


def encode_sentence(text, vocab2index, N=70):
    def pre_process_text(text):
        emoticons = [':-)', ':)', '(:', '(-:', ':))', '((:', ':-D', ':D', 'X-D', 'XD', 'xD', 'xD', '<3', '</3', ':\*',
                     ';-)',
                     ';)', ';-D', ';D', '(;', '(-;', ':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ':D', '=D',
                     '=)',
                     '(=', '=(', ')=', '=-O', 'O-=', ':o', 'o:', 'O:', 'O:', ':-o', 'o-:', ':P', ':p', ':S', ':s', ':@',
                     ':>',
                     ':<', '^_^', '^.^', '>.>', 'T_T', 'T-T', '-.-', '*.*', '~.~', ':*', ':-*', 'xP', 'XP', 'XP', 'Xp',
                     ':-|',
                     ':->', ':-<', '$_$', '8-)', ':-P', ':-p', '=P', '=p', ':*)', '*-*', 'B-)', 'O.o', 'X-(', ')-X']
        text = text.replace(".", " ").lower()
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
        users = re.findall("[@]\w+", text)
        for user in users:
            text = text.replace(user, "<user>")
        urls = re.findall(r'(https?://[^\s]+)', text)
        if len(urls) != 0:
            for url in urls:
                text = text.replace(url, "<url >")
        for emo in text:
            if emo in emoji.UNICODE_EMOJI:
                text = text.replace(emo, "<emoticon >")
        for emo in emoticons:
            text = text.replace(emo, "<emoticon >")
        numbers = re.findall('[0-9]+', text)
        for number in numbers:
            text = text.replace(number, "<number >")
        text = text.replace('#', "<hashtag >")
        text = re.sub(r"([?.!,¿])", r" ", text)
        text = "".join(l for l in text if l not in string.punctuation)
        text = re.sub(r'[" "]+', " ", text)
        return text
    tok=English()
    def tokenize(text):
        return [token.text for token in tok.tokenizer(pre_process_text(text))]
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return " ".join(map(str,encoded))


def create_k_folds_dataset(folds):
    train=pd.read_csv(f'../input/jigsaw-toxic-comment-classification-challenge/train.csv')
    #insert the kfold columns
    train['kfold'] = -1
    cat_mtpl = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
                'insult': 0.64, 'severe_toxic': 1.5, 'identity_hate': 1.5}

    for category in cat_mtpl:
        train[category] = train[category] * cat_mtpl[category]

    train['score'] = train.loc[:, 'toxic':'identity_hate'].mean(axis=1)

    train['y'] = train['score']

    #distributing the data
    kfold = KFold(n_splits = 5,shuffle=True,random_state = 42)
    for fold, (tr_i,va_i) in enumerate(kfold.split(X=train)):
        train.loc[va_i,'kfold'] = fold

    train.to_csv("../input/folds/train_folds_score_5.csv",index=False)
    print("successfully created folds")

class PreProcessJigsawDataset(object):
    def __init__(self,folds,path):
        self.folds=folds
        self.path=path
        self.tf_idf_vec=TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5), max_features = 46000)
        self.ft_vec=FastText.load('../model/jigsaw-regression-based-data/FastText-jigsaw-256D/Jigsaw-Fasttext-Word-Embeddings-256D.bin')


    def create_vectorized_dataset(self,df):
        X_list = []
        self.tf_idf_vec.fit(df['text'])
        for text in df.text:
            X_list.append(vectorizer(text,self.tf_idf_vec,self.ft_vec))
        EMB_DIM = len(self.tf_idf_vec.vocabulary_) + 256
        X_np = np.array(X_list).reshape(-1, EMB_DIM)
        X = pd.DataFrame(X_np)
        return pd.concat([X,df['y']],axis=1)
            
    def create_jigsaw_classification_dataset_folds(self):
        df = pd.read_csv(self.path+"/jigsaw-toxic-comment-classification-challenge/train.csv")
        print(df.shape)
        # Give more weight to severe toxic 
        df['severe_toxic'] = df.severe_toxic * 2
        df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
        df['y'] = df['y']/df['y'].max()

        df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
        df=self.create_vectorized_dataset(df)
        print(df.shape)
        frac_1 = 0.3
        frac_1_factor = 1.2
        
        for fld in range(self.folds):
            print(f'Fold: {fld}')
            tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                                df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                                    random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))

            tmp_df.to_csv(f'{self.path}/folds/df_fld{fld}.csv', index=False)
            print(tmp_df.shape)
            print(tmp_df['y'].value_counts())
            
            
    def create_jigsaw_clean_classification_dataset_folds(self):

        stop = stopwords.words('english')
        

        def clean(data, col):

            data[col] = data[col].str.replace(r"what's", "what is ")    
            data[col] = data[col].str.replace(r"\'ve", " have ")
            data[col] = data[col].str.replace(r"can't", "cannot ")
            data[col] = data[col].str.replace(r"n't", " not ")
            data[col] = data[col].str.replace(r"i'm", "i am ")
            data[col] = data[col].str.replace(r"\'re", " are ")
            data[col] = data[col].str.replace(r"\'d", " would ")
            data[col] = data[col].str.replace(r"\'ll", " will ")
            data[col] = data[col].str.replace(r"\'scuse", " excuse ")
            data[col] = data[col].str.replace(r"\'s", " ")

            # Clean some punctutations
            data[col] = data[col].str.replace('\n', ' \n ')
            data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
            # Replace repeating characters more than 3 times to length of 3
            data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
            # Add space around repeating characters
            data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
            # patterns with repeating characters 
            data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
            data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
            data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
            data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
            data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

            return data
        
        
        df = pd.read_csv(self.path+"/jigsaw-toxic-comment-classification-challenge/train.csv")
        print(df.shape)
        df['severe_toxic'] = df.severe_toxic * 2
        df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
        df['y'] = df['y']/df['y'].max()
        df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
        df = clean(df,'text')
        
        
        frac_1 = 0.3
        frac_1_factor = 1.2
        
        for fld in range(self.folds):
            print(f'Fold: {fld}')
            tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                                df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                                    random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))

            tmp_df.to_csv(f'{self.path}/folds/df_clean_fld{fld}.csv', index=False)
            print(tmp_df.shape)
            print(tmp_df['y'].value_counts())

        
    def create_ruddit_dataset_folds(self):
        df_ = pd.read_csv(self.path+"/ruddit-jigsaw-dataset/Dataset/ruddit_with_text.csv")
        print(df_.shape)

        df_ = df_[['txt', 'offensiveness_score']].rename(columns={'txt': 'text',
                                                                        'offensiveness_score':'y'})

        df_['y'] = (df_['y'] - df_.y.min()) / (df_.y.max() - df_.y.min()) 
        df_.y.hist()

        frac_1 = 0.7
        
        for fld in range(self.folds):
            print(f'Fold: {fld}')
            tmp_df = df_.sample(frac=frac_1, random_state = 10*(fld+1))
            tmp_df.to_csv(f'{self.path}/folds/df2_fld{fld}.csv', index=False)
            print(tmp_df.shape)
            print(tmp_df['y'].value_counts())
            
    def create_unhealthy_comments_classification_dataset_folds(self):
        df = pd.read_csv(self.path+"/unhealthy-conversations/unhealthy_full.csv")
        df=df[df._trust>0.8]
        df=df[['comment',"antagonize","condescending","dismissive","generalisation","generalisation_unfair","healthy","hostile","sarcastic"]]
        print(df.shape)
        
        # decrease toxicity if it's healthy
        df['healthy'] = df.healthy * -1
        df['y'] = (df[["antagonize","condescending","dismissive","generalisation","generalisation_unfair","healthy","hostile","sarcastic"]].sum(axis=1) ).astype(int)
        df.loc[df["y"] <0, "y"] = 0
        df['y'] = df['y']/df['y'].max()
        
        df= df[['comment', 'y']].rename(columns={'comment': 'text'})
        df=df[df["text"].astype(str).str.len()>0]
        frac_1 = 0.3
        frac_1_factor = 1.2
        
        for fld in range(self.folds):
            print(f'Fold: {fld}')
            tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                                df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                                    random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))

            tmp_df.to_csv(f'{self.path}/folds/df3_fld{fld}.csv', index=False)
            print(tmp_df.shape)
            print(tmp_df['y'].value_counts())
            
            
    
if __name__ == "__main__":
#     pre_processor=PreProcessJigsawDataset(7,"../input")
#     print(f'pre-processing toxic classification dataset')
#     pre_processor.create_jigsaw_classification_dataset_folds()
    
#     print(f'pre-processing toxic classification clean dataset')
#     pre_processor.create_jigsaw_clean_classification_dataset_folds()
    
#     print(f'pre-processing ruddit dataset')
#     pre_processor.create_ruddit_dataset_folds()
    
#     print(f'pre-processing unhealthy comments dataset')
#     pre_processor.create_unhealthy_comments_classification_dataset_folds()
    
    create_k_folds_dataset(5)
    