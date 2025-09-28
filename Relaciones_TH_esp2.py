import pandas as pd
import numpy as np
import spacy
import time
import sys

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

#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
load_vectors_in_lang(nlp,"data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv



##########################################    RELACIONES DE CONCEPTNET    ############################

#cargar relaciones para trabajar de manera local
df_diccionario = pd.read_pickle("data/Relaciones_generales.pickle")
df_diccionario_generales = df_diccionario.to_dict()

df_diccionario = pd.read_pickle("data/Relaciones_especificas.pickle")
df_diccionario_especificas = df_diccionario.to_dict()

##########################################    MÉTODOS NUEVOS      ############################
palabras_negacion_adicionales = [
    "no", "not", "n't", "never", "none", "nobody", "nowhere", "nothing", 
    "neither", "nor", "without", "cannot", "can't", "did not", "didn't", 
    "does not", "doesn't", "do not", "don't", "will not", "won't", 
    "would not", "wouldn't", "could not", "couldn't", "should not", "shouldn't", 
    "must not", "mustn't", "might not", "mightn't", "may not", "lack", 
    "absent", "fail", "deny", "refuse", "against", "opposite", "exclude", 
    "except", "prevent", "avoid", "prohibit", "ban", "restrict", "decline", 
    "reject"
]
STOP_WORDS_RT={"'d","'ll","'m","'re","'s","'ve",'a','am','an','and','are','as','at', 'be','i', 'if', 'in', 'is', 'it', 'its',
             'itself','me','my','of','or','our', 'ours', 'ourselves','so', 'than', 'that', 'the','their', 'them',
              'themselves','there', 'thereafter', 'thereby','they','this', 'those','to','thus','us','was', 'we','were',
              'you', 'your', 'yours', 'yourself', 'yourselves', '‘d', '‘ll', '‘m', '‘re', '‘s', '‘ve', '’d', '’ll', '’m',
               '’re', '’s', '’ve'}

def representacion_entidadesDavid(nlp,texto):
    dir_sust=dict()
    palabras=[]
    b=1.0
    pos=[]
    lemmas=[]
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return dir_sust,palabras,lemmas,pos
    pos=[]
    lemmas=[]
    tokens=[]
    tokenshead=[]
    tokenschild=[]
    entidades=[]
    doc =nlp(texto.lower())
    for token in doc:
        #print([child for child in token.children],token.text, token.lemma_, token.pos_,token.dep_,token.head.text,token.head.lemma_, token.head.pos_)
        if token.text == "nobody" or token.text == "one":
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ not in ["DET","ADP","AUX","ADV","ADJ","NUM","PRON"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ in ["ADJ"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if child.pos_ in ["ADP","AUX"]:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        pos.append(token.pos_)
        lemmas.append(token.lemma_)
        tokens.append(token.text)
        tokenshead.append(token.head.text)
        tokenschild.append([child for child in token.children])
    #print(entidades)
    dir_entidades=dict()
    for e in entidades:
        #print(e[2])
        if e[3] not in ['PUNCT','CCONJ']:
            if e[2] not in dir_entidades and e[2] not in ["not"]:
                if str(e[0]) in ["no"]:
                    dir_entidades[str(e[2])]=str(e[0])
                elif e[1] in ["<UNK>","DET","ADP",'CCONJ','PRON']:
                    dir_entidades[e[2]]=""
                else:
                    # if e[1] in ["NOUN"]:
                    #     dir_entidades[str(e[2])+" "+str(e[0])]=""
                    if e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=""
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=""
                    else:
                        if e[1] not in ["PRON","PUNCT"]:# segundo agregue
                            dir_entidades[e[2]]=str(e[0])
            else:
                if e[2] not in ["not"]: #checar
                    if str(e[0]) in ["no"]:
                        dir_entidades[str(e[2])]=str(dir_entidades[e[2]])+","+str(e[0])
                    # elif e[1] in ["NOUN"]:
                    #         dir_entidades[str(e[2])+" "+str(e[0])]=""
                    elif e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=""
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=""
                    #elif e[1] not in ["<UNK>","DET","ADP",'CCONJ','PRON',"PUNCT"]:
                    elif e[1] not in ["<UNK>","DET",'CCONJ','PRON',"PUNCT"]:
                        if str(dir_entidades[str(e[2])])=="":
                            dir_entidades[str(e[2])]=str(e[0])
                        else:
                            dir_entidades[str(e[2])]=str(dir_entidades[e[2]])+","+str(e[0])+","
    print(pos)
    print(lemmas)
    print(tokens)
    print(tokenshead)
    print(tokenschild)
    if len(list(dir_entidades.keys()))==0:
        ls=texto.split()
        for a in ls:
            dir_entidades[a]=""
        return dir_entidades,ls,lemmas,pos
    else:        
        return dir_entidades,list(dir_entidades.keys()),lemmas,pos

def eliminacion_espacios(lista):
    eliminar_espacios=lista.count("")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("")
    eliminar_espacios=lista.count("be")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("be")
    return lista

def found_neg(at_t):
    for t in at_t:
        if t in palabras_negacion_adicionales:
            return True,t
    return False,""

def check_entail_syn(wt,wh):
    # guardo conjuntos de sinonimos para un uso posterior
    if wt in df_diccionario_generales and wh in df_diccionario_generales:
        synt_i=df_diccionario_generales[wt]["synonym"]
        synt_i=synt_i.union(df_diccionario_especificas[wt]["synonym"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["form_of"])
        synh_i=df_diccionario_generales[wh]["synonym"]
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
        #print("---")
        #print(synt_i)
        #print(synh_i)
        if wt ==wh:                #COMPARACION DIRECTA DE wt y wh
            return True,"same"
        elif wh in synt_i:
            return True,"synonym"
        elif wt in synh_i:
            return True,"synonym"
        elif len(synt_i.intersection(synh_i))>0:
            return True,"synonym"
        else:
            return False,""
    else:
        return False,""
         
def check_entail_gen(wt,wh): # t->h
    relaciones_g1=["form_of","is_a","used_for","entails","causes","synonym","manner_of"]
    if wh in df_diccionario_especificas:
        synh_i=df_diccionario_especificas[wh]["synonym"]
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_especificas[wh]["form_of"])
        synh_i=synh_i.union(df_diccionario_especificas[wh]["is_a"])
        synh_i=synh_i.union(df_diccionario_especificas[wh]["manner_of"])
        synh_i=synh_i.union(df_diccionario_especificas[wh]["used_for"])
    else:
        synh_i=set()
    if wt in df_diccionario_generales and wh in df_diccionario_generales:
        for r_g in relaciones_g1:
            #if wh in df_diccionario_generales[wt][r_g]:
            if len(synh_i.intersection(df_diccionario_generales[wt][r_g]))>0:
                #print(wt,wh,r_g,"generales",len(synh_i.intersection(df_diccionario_generales[wt][r_g])))
                return True,r_g
                
        return False,""  # hiponimos
    else:
        return False,""

def check_contradiction_ant(wt,wh):
    relaciones_g2=["antonym","distinct_from"]
    # guardo conjuntos de sinonimos para un uso posterior
    if wt in df_diccionario_generales and wh in df_diccionario_generales:
        synt_i=df_diccionario_generales[wt]["synonym"]
        synt_i=synt_i.union(df_diccionario_especificas[wt]["synonym"])
        synt_i=synt_i.union(df_diccionario_generales[wt]["form_of"])
        synh_i=df_diccionario_generales[wh]["synonym"]
        synh_i=synh_i.union(df_diccionario_especificas[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
        for r_g in relaciones_g2:
            if wh in df_diccionario_generales[wt][r_g] or len(synh_i.intersection(df_diccionario_generales[wt][r_g]))>0:
                return True,r_g
            elif wh in df_diccionario_especificas[wt][r_g] or len(synh_i.intersection(df_diccionario_especificas[wt][r_g]))>0:
                return True,r_g
        for r_g in relaciones_g2:
            if wt in df_diccionario_generales[wh][r_g] or len(synt_i.intersection(df_diccionario_generales[wh][r_g]))>0:
                return True,r_g
            elif wt in df_diccionario_especificas[wh][r_g] or len(synt_i.intersection(df_diccionario_especificas[wh][r_g]))>0:
                return True,r_g   
        return False,""
    else:
        return False,""

def check_info_adicional(wt,wh):
    relaciones_adicionales=["part_of","at_location","has_a","has_last_subevent","has_property",
                            "defined_as","located_near","receives_action","made_of","has_subevent",
                            "has_first_subevent","has_prerequisite"]
    if wt in df_diccionario_generales and wh in df_diccionario_generales:        
        setT_ia=set()
        setH_ia=set()
        for r_g in relaciones_adicionales:
            setT_ia=setT_ia.union(df_diccionario_generales[wt][r_g])
            setH_ia=setH_ia.union(df_diccionario_especificas[wh][r_g])
        temSetT=set()
        for r_g in ["part_of"]:
            for e in setT_ia:
                if e in df_diccionario_generales:
                    temSetT=temSetT.union(df_diccionario_generales[e][r_g])
        setT_ia=setT_ia.union(temSetT)
        #print("texto",setT_ia)
        #print("hipotesis",setH_ia)
        
        intersec=(setT_ia).intersection(setH_ia)
        if len(intersec)>0:
            #print("inter",intersec)
            return True,"part_of"
        return False,""
    else:
        return False,""
    
def check_neutral_rel(wt,wh):
    relaciones_g3=["related_to","similar_to"]
    if wh in df_diccionario_generales:
        synh_i=df_diccionario_especificas[wh]["synonym"]
        synh_i=synh_i.union(df_diccionario_generales[wh]["synonym"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["form_of"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["is_a"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["manner_of"])
        synh_i=synh_i.union(df_diccionario_generales[wh]["used_for"])
    else:
        synh_i=set()
    if wt in df_diccionario_generales and wh in df_diccionario_generales:
        for r_g in ["form_of","is_a","used_for","entails","causes","synonym","manner_of"]:
            if wh in df_diccionario_especificas[wt][r_g]:
                return True,r_g
            if len(synh_i.intersection(df_diccionario_especificas[wt][r_g]))>0:
                return True,r_g
        for r_g in relaciones_g3:
            if wh in df_diccionario_generales[wt][r_g]:
                return True,r_g
        for r_g in relaciones_g3:
            if wh in df_diccionario_especificas[wt][r_g]:
                return True,r_g
        return False,""
    else:
        return False,""

def get_atributos(at_t,at_h):
    t_temp = set(eliminacion_espacios(at_t.split(",")))
    h_temp = set(eliminacion_espacios(at_h.split(",")))
    
    t_atributos =set()
    h_atributos =set()
    
    for t_ in t_temp:
        if t_ not in STOP_WORDS_RT and t_!="" and t_!=" " and t_!=",":
            t_atributos.add(t_)
    for h_ in h_temp:
        if h_ not in STOP_WORDS_RT and h_!="" and h_!=" " and h_!=",":
            h_atributos.add(h_)
    return " ".join(t_atributos)," ".join(h_atributos)

def check_atributos(at_t,at_h):
    t_temp = set(eliminacion_espacios(at_t.split(",")))
    h_temp = set(eliminacion_espacios(at_h.split(",")))
    
    t_atributos =set()
    h_atributos =set()
    
    for t_ in t_temp:
        if t_ not in STOP_WORDS_RT and t_!="" and t_!=" " and t_!=",":
            t_atributos.add(t_)
    for h_ in h_temp:
        if h_ not in STOP_WORDS_RT and h_!="" and h_!=" " and h_!=",":
            h_atributos.add(h_)
    print("atributos de T",t_atributos)
    print("atributos de H",h_atributos)
    vt,fN_t=found_neg(t_atributos)
    vh,fN_h=found_neg(h_atributos)
    if vt!=False:
        #return False,fN_t,"",0
        return False," ".join(t_atributos)," ".join(h_atributos),0
    if vh!=False:
        return False," ".join(t_atributos)," ".join(h_atributos),0
        #return False,"",fN_h,0
    # Checar cuantos atributos de h están contenidos en T
    found_att_t=[]
    found_att_h=[]
    matches=0
    for h_a in h_atributos:
        for t_a in t_atributos:
            # relaciones generales
            if h_a in t_atributos:
                matches+=1
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                break
            verificacion,tupla=check_entail_syn(t_a,h_a) #COMPARACION SINONIMOS DE wt y wh
            if verificacion:
                matches+=1
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                break
            # checo en relaciones generales si se encuentra el token de la hipótesis
            verificacion,tupla=check_entail_gen(t_a,h_a) #COMPARACION GENERAL DE wt y wh
            if verificacion:
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                matches+=1
                break
            # contradicciones
            verificacion,tupla=check_contradiction_ant(t_a,h_a) #COMPARACION CONTRA DE wt y wh
            if verificacion:
                return False,t_a,h_a,0
            #conjuntos de información adicional part_of escalar y subir
            verificacion,tupla=check_info_adicional(t_a,h_a) #COMPARACION INFO ADICIONAL DE wt y wh
            if verificacion:
                found_att_t.append(t_a)
                found_att_h.append(h_a)
                matches+=1
                break
            #especificas
            verificacion,tupla=check_neutral_rel(t_a,h_a) #COMPARACION NEUTRAL DE wt y wh
            if verificacion:
                return False,t_a,h_a,2
    if len(h_atributos)==0:
        return True, " ".join(t_atributos),"",1
    elif len(t_atributos)==0 and len(h_atributos)==0:
        return True, "","",1
    elif matches==len(h_atributos):
        return True, " ".join(found_att_t)," ".join(found_att_h),1
    elif len(t_atributos)==0 and len(h_atributos)>0:
        return False,""," ".join(h_atributos),2
    elif matches!=len(h_atributos):
        return False," ".join(t_atributos)," ".join(h_atributos),2
    else:
        return False, " ".join(t_atributos)," ".join(h_atributos),0

##########################################    CARGA  DE ARCHIVOS      ############################

#samples
prueba=pd.read_pickle("data/samples/"+sys.argv[1])
#para obtener pesos
#prueba=pd.read_pickle("data/validacion/"+sys.argv[1]) # para obtener los pesos de las votaciones

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()
clases = prueba["gold_label"].to_list()

# lista de listas para dataframe
new_data = {'Texto':[],'Hipotesis':[],'TextoL':[],'HipotesisL':[],'dicEntT':[],'dicEntH':[],
            'ConteosR':[],'ConteosG1':[],'ConteosG2':[],'ConteosG3':[],'ConteosG4':[],
            'clases' : []}

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
        new_data['TextoL'].append([])
        new_data['HipotesisL'].append([])
        new_data['dicEntT'].append([])
        new_data['dicEntH'].append([])
        new_data['ConteosR'].append([])
        new_data['ConteosG1'].append([])
        new_data['ConteosG2'].append([])
        new_data['ConteosG3'].append([])
        new_data['ConteosG4'].append([])
        new_data['clases'].append(9)
    else:
        print("Correcto")
        print(texto_i)
        r_t,t_clean_m,lemmas_t,pos_t=representacion_entidadesDavid(nlp,texto_i)
        print(r_t)

        print(hipotesis_i)
        r_h,h_clean_m,lemmas_h,pos_h=representacion_entidadesDavid(nlp,hipotesis_i)
        print(r_h)

        # lista de relaciones
        lista_rel_G1=[]
        lista_rel_G2=[]
        lista_rel_G3=[]
        lista_rel_G4=[]

        # primero evaluar si existen acronimos que se puedan identificar
        sinT=[]
        sinH=[]
        for t in lemmas_h:
            if t in df_diccionario_generales:
                sinH.append((df_diccionario_generales[t]["synonym"]).union(df_diccionario_especificas[t]["synonym"]))
        print(sinH)
        new_text=" ".join(lemmas_t) # reconstuir el texto
        words_found=[]
        print("--------------------------------------------------------")
        for e_i in range(len(sinH)):
            for e_syn in sinH[e_i]:
                if "_" in e_syn:
                    nsin=str(e_syn).replace("_"," ")
                    #print(nsin)
                    if(" "+nsin+" " in new_text):
                        #lista_rel_G1.append((lemmas_h[e_i],"synonym",nsin))
                        #words_found.append(lemmas_h[e_i])
                        for n_s in nsin.split():
                            print("se encontro primero",lemmas_h[e_i],n_s)
                            if n_s in r_t and lemmas_h[e_i] in r_h:
                                verif_att,att_t,att_h,cat=check_atributos(r_t[n_s],r_h[lemmas_h[e_i]])
                                if verif_att:
                                    lista_rel_G1.append((att_t+" "+n_s,rel_found,att_h+" "+lemmas_h[e_i]))
                                    words_found.append(lemmas_h[e_i])
                                    matching=True
                                    break
                                else:
                                    if cat==2:
                                        lista_rel_G3.append((att_h+" "+lemmas_h[e_i],"is_a",att_t+" "+n_s))
                                        words_found.append(lemmas_h[e_i])
                                        matching=True
                                        break
                                    else:
                                        lista_rel_G2.append((att_t+" "+n_s,"distinct_from",att_h+" "+lemmas_h[e_i])) ######### ESTE ES EL PROBLEMA
                                        words_found.append(lemmas_h[e_i])
                                        matching=True
                                        break
        print(new_text)
        print(words_found)

        # Matriz de alineamiento para probar la contención de las entidades

        t_vectors_n=get_matrix_rep2(t_clean_m, nlp, normed=True)
        h_vectors_n=get_matrix_rep2(h_clean_m, nlp, normed=True)

        redondeo=2
        ma_n=np.dot(t_vectors_n,h_vectors_n.T)
        ma_n = np.clip(ma_n, 0, 1).round(redondeo)
        ma=pd.DataFrame(ma_n,index=t_clean_m,columns=h_clean_m)
        print(ma)

        top_k=3
        # # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
        print("proceso lexico")
        print(ma,ma.columns)
        borrar_g=[]
        borrar_c=[]
        borrar_e=[]
        for c_c in ma.columns:
            if c_c not in words_found:
                print("columna a checar",c_c)
                # filtrar el top 3 de los mejores similitud coseno para cada token de H vs tokens de T que sean mayores a 0
                # una vez que encontremos quien se sale del ciclo
                temp=ma[c_c].sort_values(ascending=False)
                ranks=list(temp[:top_k].index)
                #valranks=list(temp[:top_k].values)
                #print(valranks,ranks)
                matching=False
                for r_i in range(len(ranks)): 
                    print("busqeuda",r_i,c_c,ranks[r_i])
                    # relaciones generales
                    verificacion,rel_found=check_entail_syn(ranks[r_i],c_c) #COMPARACION SINONIMOS DE wt y wh
                    if verificacion:
                        print("sindasdada",ranks[r_i],c_c,rel_found)
                        verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                        if verif_att:
                            lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                            matching=True
                            break
                        else:
                            if cat==2:
                                lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                                matching=True
                                break
                            else:
                                lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" "+c_c)) ######### ESTE ES EL PROBLEMA
                                matching=True
                                break
                    # checo en relaciones generales si se encuentra el token de la hipótesis
                    verificacion,rel_found=check_entail_gen(ranks[r_i],c_c) #COMPARACION GENERAL DE wt y wh
                    if verificacion:
                        print("gensadas",ranks[r_i],c_c,rel_found)
                        verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                        if verif_att:
                            lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                            matching=True
                            break
                        else:
                            if cat==2:
                                lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                                matching=True
                                break
                            else:
                                lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" b"+c_c))
                                matching=True
                                break
                    # contradicciones
                    verificacion,rel_found=check_contradiction_ant(ranks[r_i],c_c) #COMPARACION CONTRA DE wt y wh
                    if verificacion:
                        att_t,att_h=get_atributos(r_t[ranks[r_i]],r_h[c_c])
                        lista_rel_G2.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                        matching=True
                        break
                    #conjuntos de información adicional part_of escalar y subir
                    verificacion,rel_found=check_info_adicional(ranks[r_i],c_c)#COMPARACION NEUTRAL DE wt y wh
                    if verificacion:
                        verif_att,att_t,att_h,cat=check_atributos(r_t[ranks[r_i]],r_h[c_c])
                        if verif_att:
                            lista_rel_G1.append((att_t+" "+ranks[r_i],rel_found,att_h+" "+c_c))
                            matching=True
                            break
                        else:
                            if cat==2:
                                lista_rel_G3.append((att_h+" "+c_c,"is_a",att_t+" "+ranks[r_i]))
                                matching=True
                                break
                            else:
                                lista_rel_G2.append((att_t+" "+ranks[r_i],"distinct_from",att_h+" c"+c_c))
                                matching=True
                                break
                    #especificas
                    verificacion,rel_found=check_neutral_rel(ranks[r_i],c_c)
                    if verificacion:
                        lista_rel_G3.append((c_c,rel_found,ranks[r_i]))
                        matching=True
                        break
                if matching==False:
                    lista_rel_G4.append(("","unknown",c_c))
        lista_rel_ST=[]
        lista_rel_ST.extend(lista_rel_G1)
        lista_rel_ST.extend(lista_rel_G2)
        lista_rel_ST.extend(lista_rel_G3)
        lista_rel_ST.extend(lista_rel_G4)

        new_data['Texto'].append(texto_i)
        new_data['Hipotesis'].append(hipotesis_i)
        new_data['TextoL'].append(lemmas_t)
        new_data['HipotesisL'].append(lemmas_h)
        new_data['dicEntT'].append(r_t)
        new_data['dicEntH'].append(r_h)
        new_data['ConteosR'].append(lista_rel_ST[:])
        new_data['ConteosG1'].append(lista_rel_G1[:])
        new_data['ConteosG2'].append(lista_rel_G2[:])
        new_data['ConteosG3'].append(lista_rel_G3[:])
        new_data['ConteosG4'].append(lista_rel_G4[:])
        new_data['clases'].append(clases[i])

df_resultados = pd.DataFrame(new_data)
#salida de muestreos
df_resultados.to_pickle("output2/relationships/"+sys.argv[1]) #cambiar a solo numero para rapido procesamiento
#salida de pesos
#df_resultados.to_pickle("salida/validacion/"+sys.argv[1]+"_.pickle") #cambiar a solo numero para rapido procesamiento
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")