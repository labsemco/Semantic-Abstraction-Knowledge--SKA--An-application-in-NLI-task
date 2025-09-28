# Semantic-Abstraction-Knowledge--SKA--An-application-in-NLI-task
Semantic Knowledge Abstraction: Consistent Reasoning in LLMs for Natural Language Inference

Evaluation of LLMs on the NLI task (RTEGLUE, SNLI, SciTail, SICK, Diagnostic corpora)

<div align="center">
  
# 🤖 AI Models Comparison
  
</div>

<table>
  <thead>
    <tr>
      <th>Company</th>
      <th>Model</th>
      <th>ID</th>
      <th>Size</th>
      <th>Architecture</th>
      <th>Parameters</th>
      <th>Quantization</th>
      <th>Context Length</th>
      <th>Embedding Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><strong>Meta</strong></td>
      <td><code>llama3.1:latest</code></td>
      <td><code>91ab477bec9d</code></td>
      <td>4.7 GB</td>
      <td>llama</td>
      <td>8.0B</td>
      <td>Q4_0</td>
      <td>131072</td>
      <td>4096</td>
    </tr>
    <tr>
      <td><code>llama3.2:latest</code></td>
      <td><code>a80c4f17acd5</code></td>
      <td>2.0 GB</td>
      <td>llama</td>
      <td>3.2B</td>
      <td>Q4_K_M</td>
      <td>131072</td>
      <td>3072</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Google</strong></td>
      <td><code>gemma2:latest</code></td>
      <td><code>ff02c3702f32</code></td>
      <td>5.4 GB</td>
      <td>gemma2</td>
      <td>9.2B</td>
      <td>Q4_0</td>
      <td>8192</td>
      <td>3584</td>
    </tr>
    <tr>
      <td><code>gemma2:2b</code></td>
      <td><code>8ccf136fdd52</code></td>
      <td>1.6 GB</td>
      <td>gemma2</td>
      <td>2.6B</td>
      <td>Q4_0</td>
      <td>8192</td>
      <td>2304</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Microsoft</strong></td>
      <td><code>phi3:medium</code></td>
      <td><code>cf611a26b048</code></td>
      <td>7.9 GB</td>
      <td>phi3</td>
      <td>14.0B</td>
      <td>Q4_0</td>
      <td>131072</td>
      <td>5120</td>
    </tr>
    <tr>
      <td><code>phi3:latest</code></td>
      <td><code>4f2222927938</code></td>
      <td>2.2 GB</td>
      <td>phi3</td>
      <td>3.8B</td>
      <td>Q4_0</td>
      <td>131072</td>
      <td>3072</td>
    </tr>
  </tbody>
</table>
https://ollama.com/


1. Download all the corpora and unzip them into the corpus/ folder.
2. Run the cells in the processCorpus.ipynb notebook. This will create folders in data/corpus/...
3. In the data folder, run the genSamples.ipynb notebook.
4. Run the commands to execute the script Relaciones_TH_esp.py, which will search for P-H relations and create groups G1, G2, G3, and G4.
5. Once we have obtained the relations from all the examples in the corpus samples and baseline, we will call the LLMs with the script callLLM.py and callLLM_base.py

Baseline        callLLM_base.py       ->    LLMs/
Groups          callLLM.py            ->    LLMs2/
CoT             callLLM_CoT.py        ->    LLMs_AoT/
AoT             callLLM_AoT.py        ->    LLMs_AoT-KA/
AoT+Fewshots    callLLM_AoT_fs.py     ->    LLMs_AoT-KA_FS/

6. To obtain results, it is necessary to validate that the responses from the samples are complete. To do this, run the val_answers_complete.ipynb notebook to request a response again (processed/ folder) if one with the correct format is not found. If this attempt is unsuccessful, it is labeled as NA and is not taken into account for the results. This is converted into pickle files stored in the complete/ folder.

7. Once the complete samples have been obtained, the notebook group_results.ipynb is executed. 
Our statistical tests require the creation of Cross Validation, where 10 of the 13 samples are chosen to obtain accuracy and 3 are used to train the GS_DT (our proposal) and to obtain weights for the Weighted Majority Vote WMV results. We process sample 14 to generate the fewshots for AoT_fs.
The generated pickle files have the following format (answers/ folder) for each of the groups G1-G4 and combinations thereof to test new DTs:

VM: Majority vote
WVM: Weighted Majority vote
GS_DT: our proposal 

['VM', 'WVM', 'GS_DT', 'G1', 'G2', 'G3', 'G4', DT('G1', 'G2', 'G3'), DT('G1', 'G2', 'G4'), DT('G1', 'G3', 'G4'), DT('G2', 'G3', 'G4'), DT('G1', 'G2'), DT('G1', 'G3'), DT('G1', 'G4'), DT('G2', 'G3'), DT('G2', 'G4'), DT('G3', 'G4')]





