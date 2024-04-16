import pandas as pd
from transformers import BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import itertools

import streamlit as st
import random
import time

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models


# Embeddings model -------------------------------------------------------------
project_data = pd.read_csv("gs://project_description_data/three_year_reports.csv").drop(columns=["Unnamed: 0"])
embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model
descriptions = project_data[["GMPP ID Number", "Project Name", "Description / Aims"]]
embedding = embeddings_model.encode(descriptions["Description / Aims"], convert_to_tensor=False)

def compare_descriptions(new_description: str, description_embeddings, num_projects: int, descriptions):
    encoded_sentence = model.encode(new_description, convert_to_tensor=False)
    new_sentence_scores = util.cos_sim(description_embeddings, encoded_sentence)
    pairs = {}
    for i in range(descriptions.shape[0]):
        pairs[descriptions["GMPP ID Number"][i]] = new_sentence_scores[i][0].item()

    ordered = dict(sorted(pairs.items(), key=lambda item: item[1], reverse = True))
    top = dict(itertools.islice(ordered.items(), num_projects))

    return top

# RAG enabled chat model ------------------------------------------------------
def chat_with_gemini(user_query):
  vertexai.init(project="anc-pg-sbox-team-33", location="us-central1")
  model = GenerativeModel("gemini-experimental")

  top = compare_descriptions(
    new_description = user_query,
    description_embeddings = embedding,
    num_projects = 4,
    descriptions = descriptions
  )

  rag_context = ""
  for project_id in top.keys():
    mask = (descriptions["GMPP ID Number"] == project_id)
    project = descriptions[mask].reset_index(drop=True)
    project_name = project["Project Name"][0]
    project_desc = project["Description / Aims"][0]

    rag_context += project_name + '\n' + project_desc + '\n' + str(top[project_id]) + "\n\n"

  responses = model.generate_content(
      [f'''
      You are a chatbot responding to queries from project delivery professionals. You should offer advice on networking with professionals who have delivered similar large infrastructure projects across UK government. You will be provided with information to inform your responses - please only use this information when formulating your response.

      examples=[
        input_text="""I am building a new train station. What other projects like this have been delivered in the UK in the last twenty years?""",
        output_text="""Since 2000, seven major new train stations have been delivered in the UK. The average project length was seven years. One example is the upgrade to Birmingham New street station, completed in 2011. The senior responsible owner for this project is Joe Bloggs from the Department for Transport. His email address is joe.bloggs@dft.gov.uk"""
      ]

      Here are the top related projects:
      {rag_context}

      Here is the user query:
      {user_query}
       '''],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
      )
  
  for response in responses:
    yield response.text 


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# Streamlit Frontend -----------------------------------------------------------
st.title("MajorPros")

# Initialize chat history with welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to help you with your project delivery queries. Ask me anything about infrastructure projects."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I assist you today?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response in chat message container
    with st.chat_message("assistant"):
        response_generator = chat_with_gemini(prompt)
        st.write_stream(response_generator)
        full_response = "".join(list(response_generator))  # Store the full concatenated response for history
        st.session_state.messages.append({"role": "assistant", "content": full_response})