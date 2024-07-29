from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from manager import Manager
import os
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
from typing import Any, Dict, List, Optional, Union
# from dataclasses import dataclass
import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

app = FastAPI()

# origins = [
#     "http://localhost:5173",
#     " https://bob-hack.vercel.app/",
#     "http://localhost",
# ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class credentials(BaseModel):
    data_url : str
    query : str =""
    n_goals : int | None = None
    library : str = "seaborn"
    
    
model_type = "azureOAi"
llm = HF_endpoint(model_type)
llm40= HF_endpoint("azureOAi4o")   
audit = Manager()     


@app.post("/visualization")
async def visualize(input: credentials):
    input_dict = input.model_dump()
    summary = audit.summarize(
        input.data_url,
        summary_method="llmt")
    goals = audit.goals(summary,llm,n=input.n_goals)
    charts = audit.visualize_all(
        summary=summary,
        goal=goals[0],
        llm=llm,
        library=input.library)
    image_encd = charts[0]
    image_inference = audit.inference_gen(image_encd.raster,llm40)
    input_dict.update({"image_base64_raster": image_encd.raster })
    input_dict.update({"image_inference": image_inference })
    input_dict.update({"image_base64_vega": image_encd.spec })
    return input_dict
            
@app.post("/multi_visualrecomend")
async def visualize(input: credentials):
    input_dict = input.model_dump()
    summary = audit.summarize(
        input.data_url,
        summary_method="llmt")
    goals = audit.goals(summary,llm,n=input.n_goals)
    image_list = []
    inference_list = []
    for i in range(0,input.n_goals):
        charts = audit.visualize_all(
        summary=summary,
        goal=goals[i],
        llm=llm,
        library=input.library)
        image_encd = charts[0]
        image_list.append(image_encd.raster)
        image_inference = audit.inference_gen(image_encd.raster,llm40)
        inference_list.append(image_inference)
        
    input_dict.update({"image_base64_raster": image_list })
    input_dict.update({"image_inference": inference_list })
    input_dict.update({"image_base64_vega": image_encd.spec })
    return input_dict


# @app.get("/query")
# async def visualize_query(input: credentials):
#     input_dict = input.dict()
#     summary = audit.summarize(
#         input.data_url,
#         summary_method="llmt")
# #    goals = audit.goals(summary,llm,n=input.n_goals,persona=input.query)
#     try:
#         charts = audit.visualize(
#         summary=summary,
#         goal=input.query,
#         llm=llm,
#         library=input.library)
#         image_encd = charts[0]
#         input_dict.update({"image_base64":image_encd.raster })
#         return input_dict
#     except:
#         return "chart didnt prepared"

        
@app.post("/query")
async def visualize_query(input: credentials):
    input_dict = input.model_dump()
    summary = audit.summarize(
        input.data_url,
        summary_method="llmt")
    try:
        charts = audit.visualize_all(
            summary=summary,
            goal=input.query,
            llm=llm,
            library=input.library)
        print("charts function query completed")
        if not charts:
            return {"error": "No charts were generated"}
        print(charts[0].raster)
        # Ensure charts[0] exists and has the expected structure
        image_encd = charts[0]
        print("inference starting")
        image_inference = audit.inference_gen(image_encd.raster,llm40)
        input_dict.update({"image_base64_raster": image_encd.raster })
        input_dict.update({"image_inference": image_inference })
        input_dict.update({"image_base64_vega": image_encd.spec })
        return input_dict
    except Exception as e:
        return {"error": f"Chart preparation failed: {str(e)}"}






