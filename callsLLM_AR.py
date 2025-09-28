import requests
import json
import time
import pickle
import pandas as pd
import sys

ruta="output2/relationships/"+str(sys.argv[3])

# variables
model1 = sys.argv[1]
muestreos=int(sys.argv[2])
corpus=str(sys.argv[3])[:-1]
salida="LLMs/"+str(sys.argv[1])+"/"+str(sys.argv[3]) +"processed/"

####

print("INICIA TODAS LAS RELACIONES")

inicio = time.time()

lista_respuestasOllama=[]
for c in range(muestreos):
    df_t=pd.read_pickle(ruta+corpus.lower()+str(c+1)+".pickle")
    lista_respuestasOllama=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
        You are an expert in Recognition of Textual Entailment over pairs of Text and Hypothesis.
        Based on the information provided below, classify the relationship between the given Text and Hypothesis as one of the following: "Entailment", "Neutral" or "Contradiction" and give an explanation. Respond only using the template:
        {
            "Answer": "",
            "Explanation":""
        }
        Do not modify the template.

        Text and Hypothesis to Classify:
            Text: '''+strings["Texto"]+'''
            Hypothesis: '''+ strings["Hipotesis"]+'''
            
        Relations:
            '''+str(strings["ConteosR"])+'''

        Use the information provided to classify the relationship and give an explanation.'''
    
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
            print("Error",index,c+1)
            lista_respuestasOllama.append("NA")
        
    with open(salida+"rit_"+str(c+1)+"_SKE.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    #time.sleep(300)

fin = time.time()
print("Time taken:",round(fin-inicio,2)," seconds")
