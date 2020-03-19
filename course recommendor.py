import csv

# data science imports
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# utils import
from fuzzywuzzy import fuzz

class recommendation_system:
    def __init__(self):
        self.model = NearestNeighbors(algorithm='brute',metric='cosine',n_jobs=None)
    
    def content_based_recommender(self,user):
        course_ids=[]
        with open("E:\machine learning\ML PROJECTS\course recommendation system\course details.csv",'r') as course_file:
            csv_obj=csv.reader(course_file)
            for row in csv_obj:
                if len(course_ids)<3:
                    if (fuzz.partial_ratio(row[6].lower(),user.preknown.lower())) >60 :
                        if user.field==row[2]:
                            flag=False
                            if user.sub_field==row[3]:
                                flag=True
                                course_ids.append([row[0],row[1]])
                            if len(course_ids)<3 and flag==False:
                                course_ids.append([row[0],row[1]])
        if len(course_ids)==0:
            print("Oops!!No matching courses found")
            return None
        print("The courses with match with Your profile are")
        for course_name,course_id in course_ids:
            print(course_name)
        return course_ids
    
    def content_recommender(self,course_name,course_ids):
        req_sub_field=[]
        req_course_id=[]
        hash_map=dict()
        with open("E:\machine learning\ML PROJECTS\course recommendation system\course details.csv",'r') as course_file:
            csv_obj=csv.reader(course_file)
            for row in csv_obj:
                if row[0]==course_name:
                    req_sub_field.append(row[3])
        sub_field=req_sub_field[0]
        del req_sub_field
        course_idss=[]
        for i in range(0,len(course_ids)):
            course_idss.append(course_ids[i][0])
        del course_ids
        with open("E:\machine learning\ML PROJECTS\course recommendation system\course details.csv",'r') as course_file:
            csv_obj=csv.reader(course_file)
            for row in csv_obj:
                if row[3]==sub_field and row[0]!=course_name:
                    if row[1] not in course_idss:
                        req_course_id.append(row[1])
                        hash_map[row[1]]=row[0]
        #print(hash_map)
        return req_course_id,hash_map
        
    def data_extraction(self,course_name,course_ids):
        data=[]
        req_course_id,hash_map=self.content_recommender(course_name,course_ids)
        with open("E:\machine learning\ML PROJECTS\course recommendation system\course feedback score.csv",'r') as course_file:
            csv_obj=csv.reader(course_file)
            for row in csv_obj:
                if (row[0] in req_course_id):
                    data.append(row[1:50])
        data_matrix=pd.DataFrame(data)
        del data
        data_matrix=data_matrix.transpose()
        lst=data_matrix.values.tolist()
        del data_matrix
        data=pd.DataFrame(lst,columns=req_course_id)
        del lst
        return data,hash_map,req_course_id
    
    def inference(self,model,course_name,course_ids):
        data,hash_map,req_course_id=self.data_extraction(course_name,course_ids)
        data=data.transpose()
        self.model.fit(data)
        dist,ind=self.model.kneighbors(data,n_neighbors=5)
        #ic=0
        #for i in ind:
            #jc=0
            #for j in i:
                #if jc==0:
                    #print("The nearest neighbors of the course",hash_map[req_course_id[j]],"are as follows:")
                #elif jc>0:
                    #print(hash_map[req_course_id[j]],"\tdistance",dist[ic][jc])
                #jc+=1
            #ic+=1
       
        return dist,ind,hash_map,req_course_id
            
    def score_prediction(self,course_name,course_ids):
        dist,ind,hash_map,req_course_id=self.inference(self.model,course_name,course_ids)
        avg=[]
        with open("E:\machine learning\ML PROJECTS\course recommendation system\course feedback score.csv",'r') as course_file:
            csv_obj=csv.reader(course_file)
            for row in csv_obj:
                if (row[0] in req_course_id):
                   sum1=0.00
                   for k in range(1,101):
                       sum1+=float(row[k])
                   ans=sum1/100
                   avg.append(ans)
        course_guess=dict()    
        ic=0
        for i in ind:
            jc=0
            num_sum=0
            denom_sum=0
            for j in i:
                if jc>0:
                    num_sum=(1/dist[ic][jc])*avg[j]
                    denom_sum+=(1/dist[ic][jc])
                jc+=1
            ic+=1
            course_guess[hash_map[req_course_id[j]]]=(float(num_sum/denom_sum))
        print("The predicted score")
        print(course_guess)
        return course_guess
            
               
    def recommend(self,user):
        course_ids=self.content_based_recommender(user)
        if len(course_ids)!=0:
            for i in range(0,len(course_ids)):
                course_guess=self.score_prediction(course_ids[i][0],course_ids)
                print("Some other courses which you can later try are:(recommendations for the course",course_ids[i][0],")")
                n_recommendations=3
                i=1
                for key,value in sorted(course_guess.items(),key=lambda item:item[1],reverse=True):
                    print(key)
                    if i==n_recommendations:
                        break
                    i+=1
                
class User:
    
    def get_details(self):
        self.name=input("Please enter your name : ")
        self.age=input("enter your age : ")
        self.gender=input("enter your gender : ")
        self.preknown=input("Enter the courses which you already know(enter comma between courses)")
        self.field=input("Enter the field which you are interested in")
        self.sub_field=input("Enter the sub-field that you are interested in")
        self.toughness=int(input("Enter how much tough course you can try(out of 10)"))
        
           
if __name__ == '__main__':
    print("\t\t WELCOME TO COURSE RECOMMENDATION SYSTEM")
    user=User()
    user.get_details()
    recommender = recommendation_system()
    recommender.recommend(user)