import requests
import json
import random
import time
import pickle
import pandas as pd
import sys

ruta="output2/relationships/"+str(sys.argv[3])

# modelo y donde se almacena los resultados
model1 = sys.argv[1]
muestreos=int(sys.argv[2])
salida="LLMs_AoT/"+str(sys.argv[1])+"/"+str(sys.argv[3])+"processed/" #

####

print("INICIA CALLS LLMS CON AOT")

inicio = time.time()
print(inicio)

lista_respuestasOllama=[]
for c in range(muestreos):
    df_t=pd.read_pickle(ruta+str(sys.argv[3])[:-1].lower()+str(c+1)+".pickle")
    lista_respuestasOllama=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
                You are an expert in Recognizing Textual Entailment over pairs of Premise and Hypothesis.
        Classify the relationship between the given Premise and Hypothesis as one of the following: "Entailment", "Neutral" or "Contradiction". 

        Premise: '''+strings["Texto"]+'''
        Hypothesis: '''+ strings["Hipotesis"]+'''

        Let's think step by step

        Step 1: Identify all the relationships between terms from the premise to the hypothesis. This process is performed for each of the terms in the hypothesis.
        Use the next format for relationships: (p_i,rel,h_j) where p_i is a term from the premise, h_j is a term from the hypothesis, and rel is the relation linking these terms. 
        If any terms of the hypothesis with an unknown relationship with terms in the premise, identify them as (,unknown,h_k) where h_k is in the hypothesis. 
        List the relationships found.

        Step 2: Align all the relationships found with the NLI labels: Entailment, Neutral, Contradiction; classifying them into groups G1, G2, G3 and G4 according to the following:
        G1: will contain the list of relationships that align with the entailment label
        G2: will contain the list of relationships that align with the contradiction label 
        G3: will contain the list of relationships that align with the neutrality label
        G4: will contain the list of terms of the hypothesis with an unknown relationship with the premise. 

        Step 3: Analyze each group of relationships and decide on the correct label for the premise and hypothesis presented.

        Respond only using the template:
        {
        "G1":[],
        "G2":[],
        "G3":[],
        "G4":[],
        "Answer": "",
        "Explanation": ""
        }
        Fill out the template.
            '''
    
    
        # print(prompt)
        # break
        data = {
            "prompt": prompt,
            "model": model1,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=90)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index,c)
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
        
    with open(salida+"ritB_AoT"+str(c+1)+".pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    time.sleep(300)

fin = time.time()
print(fin)
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")
