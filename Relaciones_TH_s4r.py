import pandas as pd
import numpy as np
import spacy
import time
import sys
import re

#########################    CARGA DE MODELO SPACY NLP Y WORDS EMBEDDINGS     #######################
nlp = spacy.load("en_core_web_md") # modelo de nlp

# Load vectors from dict
def load_vectors_as_dict(path):
    vectors = {}
    with open(path, 'r', encoding="utf8") as f:
        line = f.readline()
        while line:
            # Split on white spaces
            line = line.strip().split(' ')
            if len(line) > 2:
                vectors[line[0]] = np.array([float(l) for l in line[1:]], dtype=np.float32)
            line = f.readline()
    return vectors

# Load vectors in a spacy nlp
def load_vectors_in_lang(nlp, vectors_loc):
    wv= load_vectors_as_dict(vectors_loc)
    nlp.wv = wv

    # # Check if list of oov vectors exists
    # # If so, load, if not, create
    # oov_path,ext = os.path.splitext(vectors_loc)
    # oov_path = oov_path+'.oov.txt'
    # if os.path.exists(oov_path):
    #     nlp.oov = np.loadtxt(oov_path)
    # else:
    fk = list(wv.keys())[0]
    nf = wv[fk].shape[0]
    nlp.oov = np.random.normal(size=(100,nf))
    return 

def get_vector2(w, nlp, nf=300):
    if str(w) in nlp.wv:
        v = nlp.wv[str(w)]
    else: 
        v = np.zeros((1,300))[0]
        #v = np.ones((1,300))[0]
    return v.astype(np.float32)

def get_matrix_rep2(words,nlp, normed=True):
    vecs = np.array([get_vector2(w,nlp) for w in words], dtype=np.float32)
    if len(vecs) == 0:
        vecs = np.ones((1,300), dtype=np.float32)

    # Normalize vectors if desired
    if normed:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs /= norms
    return vecs

# Some cleaning especially with respect to weird punctuation
def clean_text(s):
    s = re.sub("([.,!?()-])", r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

def get_words(text, nlp, pos_to_remove=['PUNCT'], normed=True,
    lemmatize=True):

    text = clean_text(str(text)).lower()    

    # Process the document via Spacy's nlp
    doc = nlp(text)

    # Lemmatize if desired
    if lemmatize:
        text = ' '.join([w.lemma_ for w in doc])
        doc = nlp(text)

    # Get processed words removing undesired POS
    words=[]
    for w in doc:
        if w.pos_ not in pos_to_remove:
            words.append(w.text)

    return words
#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
load_vectors_in_lang(nlp,"data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv

##########################################    RELACIONES DE CONCEPTNET    ############################

#cargar relaciones para trabajar de manera local
df_diccionario = pd.read_pickle("data/Relaciones_generales.pickle")
df_diccionario_generales = df_diccionario.to_dict()

df_diccionario = pd.read_pickle("data/Relaciones_especificas.pickle")
df_diccionario_especificas = df_diccionario.to_dict()

##########################################    MÉTODOS NUEVOS      ############################

rel_concept=['synonym', 
             'antonym']

def check_relationships(wt,wh):
    for r in rel_concept:
        if wt in df_diccionario_generales:
            if wh in df_diccionario_generales[wt][r]:
                return True,(wt,r,wh)
        elif wh in df_diccionario_especificas:
            if wt in df_diccionario_especificas[wh][r]:
                return True,(wh,r,wt)
    return False,""

def get_4relationships(wt):
    a=[]
    for r in rel_concept:
        if wt in df_diccionario_generales:
            if len(df_diccionario_generales[wt][r])>0:
                a.append((wt,r,df_diccionario_generales[wt][r].pop()))
        elif wt in df_diccionario_especificas:
            if len(df_diccionario_especificas[wt][r])>0:
                a.append((wt,r,df_diccionario_especificas[wt][r].pop()))   
    for r in ['is_a']:
        if wt in df_diccionario_generales:
            if len(df_diccionario_generales[wt][r])>0:
                a.append((wt,r,df_diccionario_generales[wt][r].pop()))
        if wt in df_diccionario_especificas:
            if len(df_diccionario_especificas[wt][r])>0:
                a.append((wt,r,df_diccionario_especificas[wt][r].pop()))        
    if len(a)>0:
        return True,a
    else:
        return False,a

##########################################    CARGA  DE ARCHIVOS      ############################

prueba=pd.read_pickle("data/samples/"+sys.argv[1]) 

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()
clases = prueba["gold_label"].to_list()

# lista de listas para dataframe
new_data = {'Texto':[],'Hipotesis':[], 'ConteosR':[],'clases' : []}

##########################################    INICIO DE PROCESO      ############################
inicio = time.time()
for i in range(len(textos)):
#for i in range(4):free
    print(i)
    texto_i=str(textos[i])
    hipotesis_i=str(hipotesis[i])
       
    #Revisar si es numerico la hipótesis para identificar en los resultados
    # para hacer el proceso o no
    if(type(hipotesis[i])==type(1.0) or type(textos[i])==type(1.0)):
        print("Falla")
        new_data['Texto'].append(texto_i)
        new_data['Hipotesis'].append(hipotesis_i)
        new_data['ConteosR'].append([])
        new_data['clases'].append(9)
    else:
        print("Correcto")
        print(texto_i)
        #t_i=list(set(texto_i.split()))
        t_i=list(set(get_words(texto_i, nlp)))

        print(hipotesis_i)
        #h_i=list(set(hipotesis_i.split()))
        h_i=list(set(get_words(hipotesis_i, nlp)))

        # Matriz de alineamiento para probar la contención de las entidades

        t_vectors_n=get_matrix_rep2(t_i, nlp, normed=True)
        h_vectors_n=get_matrix_rep2(h_i, nlp, normed=True)

        redondeo=2
        ma_n=np.dot(t_vectors_n,h_vectors_n.T)
        ma=pd.DataFrame(ma_n,index=t_i,columns=h_i)
        print(ma)

        top_k=1
        # # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPTNET
        print("proceso lexico")
        print(ma,ma.columns)

        lista_rel_ST=[]

        for c_c in ma.columns:
            found,r_s = get_4relationships(c_c)
            if (found):
                print("encontró",r_s)
                lista_rel_ST.extend(r_s)
        for c_c in ma.index:
            found,r_s = get_4relationships(c_c)
            if (found):
                print("encontró",r_s)
                lista_rel_ST.extend(r_s)
        
        new_data['Texto'].append(texto_i)
        new_data['Hipotesis'].append(hipotesis_i)
        new_data['ConteosR'].append(lista_rel_ST[:])
        new_data['clases'].append(clases[i])

df_resultados = pd.DataFrame(new_data)
df_resultados.to_pickle("output2/relationships/"+sys.argv[1]+"allRelationsDR_4r.pickle") #cambiar a solo numero para rapido procesamiento
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")