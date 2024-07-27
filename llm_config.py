import getpass
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from typing import Any, Dict, List, Optional, Union


def HF_endpoint(producer):
    if producer=="hf":
        print("Using Huggingface model")
        repo_id="mistralai/Mistral-7B-Instruct-v0.2"
        llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=None,temperature=0.7,token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        return llm
    elif producer == "azureOAi":
        print("Using Azure Model")
        os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        llm = AzureChatOpenAI(
            azure_deployment="gpt-35-audit",
            api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
            )
        return llm
    else:
        print("Producer parameter missing or some other error at config level")
        
