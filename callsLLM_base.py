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
salida="LLMs/"+str(sys.argv[1])+"/"+str(sys.argv[3])+"processed/" #+"validacion/" # le agregue la ruta donde lo pondra validacion/

####

inicio = time.time()

for c in range(muestreos):
    df_t=pd.read_pickle(ruta+str(sys.argv[3])[:-1].lower()+str(c+1)+".pickle")
    lista_respuestasOllama=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = '''
        You are an expert in Recognition of Textual Entailment over pairs of Premise and Hypothesis.
        Classify the relationship between the given Premise and Hypothesis as one of the following: "Entailment", "Neutral" or "Contradiction" and give an explanation. Respond only using the template:
        {
            "Answer": "",
            "Explanation":""
        }
        Do not modify the template.
        
        Premise and Hypothesis to Classify:
            Premise: '''+strings["Texto"]+'''
            Hypothesis: '''+ strings["Hipotesis"]
    
        #print(prompt)
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
            lista_respuestasOllama.append("NA")
        
    with open(salida+"rit_Base_"+str(c+1)+".pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
#        time.sleep(200)
    
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")
