# IMPORT
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
import time
import gc 
from IPython.display import display, HTML
from pprint import pprint
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_colwidth=300


# Seed
pl.seed_everything(seed=42)


class JigsawRidgeModel:
    def __init__(self,path,folds):
        self.path=path
        self.df_val=pd.read_csv(f"{self.path}/jigsaw-toxic-severity-rating/validation_data.csv")
        self.df_sub=pd.read_csv(f"{self.path}/jigsaw-toxic-severity-rating/comments_to_score.csv")
        self.folds=folds
        
    def train_and_predict(self,file_prefix):
        start_time = time.time()
        val_preds_arr1 = np.zeros((self.df_val.shape[0], self.folds))
        val_preds_arr2 = np.zeros((self.df_val.shape[0], self.folds))
        test_preds_arr = np.zeros((self.df_sub.shape[0], self.folds))

        for fld in tqdm(range(self.folds)):
            print("\n\n")
            print(f' ****************************** FOLD: {fld} ******************************')
            df = pd.read_csv(f'{self.path}/folds/{file_prefix}{fld}.csv')
            print(df.shape)

            features = FeatureUnion([
                ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),

            ])

            pipeline = Pipeline(
                [
                    ("features", features),
                    ("clf", Ridge())
                ]
            )
            # Train the pipeline
            pipeline.fit(df['text'].values.astype(str), df['y'])

            feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                          np.round(pipeline['clf'].coef_,2) )), 
                                 key = lambda x:x[1], 
                                 reverse=True)
            val_preds_arr1[:,fld] = pipeline.predict(self.df_val['less_toxic'])
            val_preds_arr2[:,fld] = pipeline.predict(self.df_val['more_toxic'])
            test_preds_arr[:,fld] = pipeline.predict(self.df_sub['text'])
            
        print("\n--- %s seconds ---" % (time.time() - start_time))
        return val_preds_arr1,val_preds_arr2, test_preds_arr
    
    
if __name__=="__main__":
    trainer=JigsawRidgeModel("../input",7)
    
    # jigsaw classification dataset
    val_preds_arr1,val_preds_arr2, test_preds_arr=trainer.train_and_predict("df_fld")
    d1=pd.DataFrame(val_preds_arr1)
    d1.to_csv(f'../output/predictions/df1_1.csv', index=False)
    d2=pd.DataFrame(val_preds_arr2)
    d2.to_csv(f'../output/predictions/df1_2.csv', index=False)
    d3=pd.DataFrame(test_preds_arr)
    d3.to_csv(f'../output/predictions/df1_3.csv', index=False)

    
    # jigsaw clean classification dataset
    val_preds_arrc1,val_preds_arrc2, test_preds_arrc=trainer.train_and_predict("df_clean_fld")
    d1=pd.DataFrame(val_preds_arrc1)
    d1.to_csv(f'../output/predictions/df2_1.csv', index=False)
    d2=pd.DataFrame(val_preds_arrc2)
    d2.to_csv(f'../output/predictions/df2_2.csv', index=False)
    d3=pd.DataFrame(test_preds_arrc)
    d3.to_csv(f'../output/predictions/df2_3.csv', index=False)
    
    
    # jigsaw ruddit dataset
    val_preds_arr1r,val_preds_arr2r, test_preds_arrr=trainer.train_and_predict("df2_fld")
    d1=pd.DataFrame(val_preds_arr1r)
    d1.to_csv(f'../output/predictions/df3_1.csv', index=False)
    d2=pd.DataFrame(val_preds_arr2r)
    d2.to_csv(f'../output/predictions/df3_2.csv', index=False)
    d3=pd.DataFrame(test_preds_arrr)
    d3.to_csv(f'../output/predictions/df3_3.csv', index=False)
    
    # jigsaw unhealthy comments dataset
    val_preds_arr1u,val_preds_arr2u, test_preds_arru=trainer.train_and_predict("df3_fld")
    d1=pd.DataFrame(val_preds_arr1u)
    d1.to_csv(f'../output/predictions/df4_1.csv', index=False)
    d2=pd.DataFrame(val_preds_arr2u)
    d2.to_csv(f'../output/predictions/df4_2.csv', index=False)
    d3=pd.DataFrame(test_preds_arru)
    d3.to_csv(f'../output/predictions/df4_3.csv', index=False)
    
    
    
    
    
