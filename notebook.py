#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import time
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


# In[127]:


static_profile_info = ''''Nursulu Sagimbayeva. 
Munich/ Saarbrücken, Germany. Open to relocation. 
Profile: I am a Master’s student in my last year, with background in the domains of NLP, Data Science, Societal Computing, and Mechanistic Interpretability. I am looking for an internship to gain insight in the industry while working on challenging NLP/AI topics.
Work experience
•Internship Financial Assets & Solutions Data Analytics November 2024-May 2025\nMunich RE Munich, Germany.
•Research Assistant June 2023-Dec 2024. Interdisciplinary Institute of Societal Computing, Saarland Informatik Campus Saarbrücken, Germany.
•Technical Content Writer July 2022-July 2023 at Hasty.ai (CloudFactory) Berlin, Germany (remote)
Education
•M.Sc. in Natural Language Processing, current GPA: 1.5 (best: 1.0) 
October 2022-currently\nSaarland University Saarbrücken, Germany
•B.A. in Translation Studies, GPA: 3.75/4.0, German: 1.38 2018-2022
Al-Farabi Kazakh National University Almaty, Kazakhstan
'''


# ### Extract PDF data 

# In[9]:


from pypdf import PdfReader


# In[10]:


# Open and read the PDF
pdf_path = "CV_2025.pdf"
reader = PdfReader(pdf_path)

# Extract text from all pages
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


# In[12]:


text


# In[13]:


metadata = reader.metadata


# In[14]:


metadata


# TODO: try to extract metadata about different CV parts

# ### Get the embeddings

# In[15]:


from huggingface_hub import login

login("") # Your API Key


# Explore other SentenceTransformer models:
# https://huggingface.co/models

# In[16]:


embedding_model = SentenceTransformer("BAAI/bge-m3")


# In[17]:


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()


# ### Chunk the data

# In[25]:


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', '– ', '• ', ' ', ''], #order matters
    chunk_size=150,
    chunk_overlap=50
)

# Create chunks of the document content
chunks = []
last_key = 0
for doc in tqdm([text]):
    doc_chunks = text_splitter.split_text(doc)
    for j in range(len(doc_chunks)):
        chunks.append({f'id': f"id{last_key}",
                       'content': doc_chunks[j],
                       'embedding': get_embedding(doc_chunks[j]),
                       })
        last_key += 1
 


# In[26]:


[el['content'] for el in chunks]


# #### Create a vector base

# In[27]:


client = chromadb.PersistentClient(path="chroma_tmp", settings=chromadb.Settings(allow_reset=True))
client.reset()


# In[83]:


cv_collection = client.create_collection(
    name="CV_2025",
    metadata={"hnsw:space": "cosine"}
)


# In[85]:


for i in range(len(chunks)):
    cv_collection.add(
        documents=chunks[i]['content'],
        ids=chunks[i]["id"],
        embeddings=chunks[i]["embedding"]
    )


# ### Get data about job postings

# In[32]:


import requests


# In[33]:


url = "https://www.arbeitnow.com/api/job-board-api"

payload={}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)


# In[40]:


response = json.loads(response.text)


# In[45]:


response.keys()


# In[46]:


response['links']


# ### Step 1: find most suitable recommendations

# In[49]:


response['data'][0]


# In[47]:


response['data'][0]['description']


# In[55]:


# Split the document into chunks_jobs
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', '– ', '• ', ' ', ''], #order matters
    chunk_size=1000,
    chunk_overlap=50
)

# Create chunks_jobs of the document content
chunks_jobs = []
last_key = 0
for doc in tqdm(response['data']):
    content = doc['description']
    doc_chunks_jobs = text_splitter.split_text(content)
    for j in range(len(doc_chunks_jobs)):
        chunks_jobs.append({f'id': f"id{last_key}",
                       'content': doc_chunks_jobs[j],
                       'embedding': get_embedding(doc_chunks_jobs[j]),
                       'metadata': {'location': doc['location'], 'remote': doc['remote'],
                                    'job_types': str(doc['job_types']), 'title': doc['title'],
                                    'company_name': doc['company_name'], 'url': doc['url'],
                                    'tags': str(doc['tags'])}
                       })
        last_key += 1
 


# In[92]:


# Create chunks_jobs_unsplit version of the document content
chunks_jobs_unsplit = []
last_key = 0
for i, doc in tqdm(enumerate(response['data'])):
    content = doc['description']
    chunks_jobs_unsplit.append({f'id': f"id{last_key}",
        'content': content,
        'embedding': get_embedding(content),
        'metadata': {'location': doc['location'], 'remote': doc['remote'],
                    'job_types': str(doc['job_types']), 'title': doc['title'],
                    'company_name': doc['company_name'], 'url': doc['url'],
                    'tags': str(doc['tags'])}
                       })
    last_key += 1
 


# ### Unsplit version 

# In[93]:


job_collections_unsplit = client.create_collection(
    name="job_collections_unsplit",
    metadata={"hnsw:space": "cosine"}
)


# In[94]:


for i in tqdm(range(len(chunks_jobs_unsplit))):
    job_collections_unsplit.add(
        documents=chunks_jobs_unsplit[i]['content'],
        ids=chunks_jobs_unsplit[i]["id"],
        metadatas=chunks_jobs_unsplit[i]["metadata"],
        embeddings=chunks_jobs_unsplit[i]["embedding"]
    )


# ### Split version

# In[56]:


job_collections = client.create_collection(
    name="jobs_collections",
    metadata={"hnsw:space": "cosine"}
)


# In[57]:


for i in tqdm(range(len(chunks_jobs))):
    job_collections.add(
        documents=chunks_jobs[i]['content'],
        ids=chunks_jobs[i]["id"],
        metadatas=chunks_jobs[i]["metadata"],
        embeddings=chunks_jobs[i]["embedding"]
    )


# In[58]:


cv_embeddings = get_embedding(text)


# In[95]:


top_results = job_collections_unsplit.query(
    query_embeddings=cv_embeddings,
    # query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    # where={"metadata_field": "is_equal_to_this"},
    # where_document={"$contains":"search_string"}
)


# In[96]:


top_results


# Add search for the metadata (internship, WS, professional, etc.)

# ### Step 2: generate cover letter

# Iterate over all top-k jobs and extract most relevant chunks

# In[63]:


cv_chunks = chunks


# In[97]:


top_job_ids = top_results['ids'][0]


# In[98]:


top_job_descriptions = job_collections_unsplit.get(
    ids=top_job_ids
)['documents']


# In[109]:


top_job_matches = job_collections_unsplit.get(
    ids=top_job_ids
)


# In[110]:


job_to_apply_to = [f"Company name: {top_job_matches['metadatas'][i]['company_name']}\nJob title: {top_job_matches['metadatas'][i]['title']}\nJob type: {top_job_matches['metadatas'][i]['job_types']}\nJob description: {top_job_matches['documents'][i]}" for i in range(len(top_job_matches['ids']))]


# In[112]:


top_cv_matches = cv_collection.query(
    query_embeddings=[get_embedding(el) for el in top_job_descriptions],
    # query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    # include=['embeddings', 'documents', 'metadatas']
    # where={"metadata_field": "is_equal_to_this"},
    # where_document={"$contains":"search_string"}
)


# In[100]:


for i in range(len(top_job_descriptions)):
    print('Job description:')
    print(top_job_descriptions[i])
    print("Best CV matches:")
    print(top_cv_matches['documents'][i])


# ### Step 3: call the model and generate the letter.

# In[103]:


from huggingface_hub import InferenceClient
from openai import OpenAI


# In[102]:


hf_token = "YOUR_HF_TOKEN"


# In[119]:


instruction = '''You are a job application assistant. Your task is, given the job description and highlights from the candidate's CV, write a cover letter tailored to the job description to maximize the chances of the candidate to get an interview invitation.
'''


# In[120]:


content = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n

### Instruction:
  {}

  ### Job description:
  {}

  ### Candidate profile:
  {}
  
  ### Candidate CV highlights:
  {}
  
  ### Response:
  '''


# In[121]:


job_to_cover_letter = dict()


# In[ ]:


client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/",
	api_key="YOUR_API_KEY"
)

for i in tqdm(range(len(top_job_descriptions))):
    messages = [
        {
            "role": "user",
            "content": content.format(instruction, job_to_apply_to[i], static_profile_info, str(top_cv_matches['documents'][i]))
        }]
    
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        messages=messages, 
    )
    
    job_to_cover_letter[job_to_apply_to[i]] = completion.choices[0].message.content
    with open('model_responses/cover_letters.json', 'w', encoding='utf-8') as file:
        json.dump(job_to_cover_letter, file)

	


# In[130]:


completion.choices[0].message.content


# In[ ]:


# vector_collections.count()

