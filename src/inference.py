
import pandas as pd
import numpy as np

class JigsawInference:
    def __init__(self,path):
        self.path=path
        self.df1_1=pd.read_csv(f"{self.path}/df1_1.csv").to_numpy()
        self.df1_2=pd.read_csv(f"{self.path}/df1_2.csv").to_numpy()
        self.df1_3=pd.read_csv(f"{self.path}/df1_3.csv").to_numpy()
                
        self.df2_1=pd.read_csv(f"{self.path}/df2_1.csv").to_numpy()
        self.df2_2=pd.read_csv(f"{self.path}/df2_2.csv").to_numpy()
        self.df2_3=pd.read_csv(f"{self.path}/df2_3.csv").to_numpy()
                
        self.df3_1=pd.read_csv(f"{self.path}/df3_1.csv").to_numpy()
        self.df3_2=pd.read_csv(f"{self.path}/df3_2.csv").to_numpy()
        self.df3_3=pd.read_csv(f"{self.path}/df3_3.csv").to_numpy()  
        
        self.df4_1=pd.read_csv(f"{self.path}/df4_1.csv").to_numpy()
        self.df4_2=pd.read_csv(f"{self.path}/df4_2.csv").to_numpy()
        self.df4_3=pd.read_csv(f"{self.path}/df4_3.csv").to_numpy()
        
        self.df_sub=pd.read_csv(f"../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
    
    def print_accuracy_and_find_weightage(self):
        p1 = self.df1_1.mean(axis=1)
        p2 = self.df1_2.mean(axis=1)
        print(f'Toxic data accuracy { np.round((p1 < p2).mean() * 100,2)}')

        p3 = self.df2_1.mean(axis=1)
        p4 = self.df2_2.mean(axis=1)
        print(f'Toxic clean data accuracy { np.round((p3 < p4).mean() * 100,2)}')
        
        p5 = self.df3_1.mean(axis=1)
        p6 = self.df3_2.mean(axis=1)
        print(f'Ruddit data accuracy { np.round((p5 < p6).mean() * 100,2)}')
        
        p7 = self.df4_1.mean(axis=1)
        p8 = self.df4_2.mean(axis=1)
        print(f'Unhealthy comments data accuracy { np.round((p7 < p8).mean() * 100,2)}')

        wts_acc = []
        for i in range(1,100):
            for j in range(1,100):
                for k in range (1, 100):
                        w1,w2,w3 = i/100,j/100,k/100
                        w4=(100-i-j-k)/100
                        if w4<0:
                            continue
                        p1_wt = w1*p1 + w2*p3 + w3*p5 + w4*p7
                        p2_wt = w1*p2 + w2*p4 + w3*p6 + w4*p8
                        wts_acc.append( (w1,w2,w3,w4, 
                                         np.round((p1_wt < p2_wt).mean() * 100,2))
                                      )
        best_values=sorted(wts_acc, key=lambda x:x[4], reverse=True)[0]
        print(best_values)

        
if __name__=="__main__":
    infer=JigsawInference("../output/predictions")
    # infer.print_accuracy_and_find_weightage()
    w1,w2,w3,w4=0.59, 0.02, 0.33, 0.06
    infer.df_sub['score'] = w1*infer.df1_3.mean(axis=1) + w2*infer.df2_3.mean(axis=1) + w3*infer.df3_3.mean(axis=1) + w4*infer.df4_3.mean(axis=1)
    infer.df_sub[['comment_id', 'score']].to_csv("../output/predictions/submission.csv", index=False)