from scaffold import ChartScaffold
import json
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
from typing import Any, Dict, List, Optional, Union
import getpass
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import base64
from llm_config import HF_endpoint

class VizInference():
    """ Generate Inference for given visual graph image"""
    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()    
    
    def genrate(self,image_b64,llm):
        
        messages =  [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are an Experienced Bank auditor and a great Data analyst who is able to extract inference, insights and information from Visual Graphs for auditing process and create a inference report  "
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{image_b64}"
          }
        },
        {
          "type": "text",
          "text": "give inference for this visual graph usefull for auditing process? dont include the recomendations regarding data entry  "
        }
      ]
    }]
        response = llm.invoke(messages)
        response = response.content
        return response
        