import requests
import json
import time
import pickle
import pandas as pd
import sys

ruta="output2/relationships/"+str(sys.argv[3])

definitionsE=["Group 1 (G1): Triplets of Generality or Equivalence. These relations usually correspond to Entailment. triplets: (pi,rel,hj) where pi is in Premise, hj is in Hypothesis and rel is the relation between pi and hj. e.g., (dog,is_a,animal) dog is in Premise, animal is in Hypothesis and relation is general between dog - animal.",
        "Group 2 (G2): Triplets of Contradictory. These often indicate Contradiction. triplets: (pi,rel,hj) where pi is in Premise, hj is in Hypothesis and rel is the relation between ti and hj. e.g., (dog,distinct_from,cat) dog is in Premise, cat is in Hypothesis and relation is distinct_from between dog - cat.",
        "Group 3 (G3): Triplets of Concretization, Specificity, or Contextuality. These typically correspond to Neutral. triplets: (hj, rel, pi), where ti is in Premise, hj is in Hypothesis, and rel is the relationship between pi and hj. For example, (dog, is_a, animal): animal is in Premise, dog is in Hypothesis, and the relationship is specific between dog - animal.",
        "Group 4 (G4): Relations not identified or categorized."]

gs=["G1","G2","G3","G4"]

grupos=["ConteosG1","ConteosG2","ConteosG3","ConteosG4"]

# mvariables
model1 = sys.argv[1]
muestreos=int(sys.argv[2])
salida="LLMs_SKA/"+str(sys.argv[1])+"/"+str(sys.argv[3])+"processed/"

####

inicio = time.time()

lista_respuestasOllama=[]
for c in range(muestreos):
    df_t=pd.read_pickle(ruta+str(sys.argv[3])[:-1].lower()+str(c+1)+".pickle")
    for f_ in range(len(definitionsE)):
        lista_respuestasOllama=[]
        for index,strings in df_t.iterrows():
            #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
            prompt = '''
            You are an expert in Recognition of Textual Entailment over pairs of Premise and Hypothesis.
            Based on the information provided below, classify the relationship between the given Premise and Hypothesis as one of the following: "Entailment", "Neutral" or "Contradiction" and give an explanation. Respond only using the template:
            {
                "Answer": "",
                "Explanation":""
            }
            Do not modify the template.

            Background Information:
            
                Word Relationship Groups:
                    '''+str(definitionsE[f_])+'''

            Premise and Hypothesis to Classify:
                Premise: '''+strings["Texto"]+'''
                Hypothesis: '''+ strings["Hipotesis"]+'''
                
            Relations:
                '''+str(gs[f_])+''' : '''+str(strings[grupos[f_]])+'''                    

            Use the information provided to classify the relationship and give an explanation.'''
        
            #print(prompt)
            data = {
                "prompt": prompt,
                "model": model1,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0,
                    "num_ctx":4096},
            }
            try:
                response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=90)        
                json_data = json.loads(response.text)
                lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
                print(index)
            except:
                print("Error",index,c)
                dat=response.text
                resp={}
                if('\"Answer\": \"Neutral\"' in dat or '\\"Answer\\": \\"Neutral\\"' in dat):
                    resp["Answer"]="Neutral"
                    #lista_respuestasOllama.append(json.dumps(json.loads("{'Answer':'Neutral','Explanation':'"+dat+"'}"),indent=2))
                elif('\"Answer\": \"Entailment\"' in dat or '\\"Answer\\": \\"Entailment\\"' in dat):
                    resp["Answer"]="Entailment"
                    #lista_respuestasOllama.append(json.dumps(json.loads("{'Answer':'Entailment','Explanation':'"+dat+"'}"),indent=2))
                elif('\"Answer\": \"Contradiction\"' in dat or '\\"Answer\\": \\"Contradiction\\"' in dat):
                    resp["Answer"]="Contradiction"
                    #lista_respuestasOllama.append(json.dumps(json.loads("{'Answer':'Contradiction','Explanation':'"+dat+"'}"),indent=2))
                else:
                    resp["Answer"]="NA"
                    #lista_respuestasOllama.append("NA")
                resp["Explanation"]=dat
                lista_respuestasOllama.append(json.dumps(resp))
            
        with open(salida+"rit_"+str(c+1)+"_"+str(gs[f_])+".pickle", "wb") as f:
            pickle.dump(lista_respuestasOllama, f)
        #time.sleep(200)
    
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")
