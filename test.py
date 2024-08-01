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
model_type = "azureOAi"
llm = HF_endpoint(model_type)

cars_data_url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"

def test_summarizer():
    summary_no_enrich = Manager.summarize(
        cars_data_url,
        summary_method="default")
    summary_enrich = Manager.summarize(cars_data_url,
                                    summary_method="llmt")
    print("Summar 1:-"+"\n")
    print(summary_no_enrich)
    print("\n"+"Summar 2:-"+"\n")
    print(summary_enrich)

    # assert summary_no_enrich != summary_enrich
    # assert "dataset_description" in summary_enrich and len(
    #     summary_enrich["dataset_description"]) > 0

sandesh = Manager()

def test_goals():
    summary = sandesh.summarize(
        cars_data_url,
         summary_method="default")

    goals = sandesh.goals(summary,llm,n=2)
    assert len(goals) == 2
    assert len(goals[0].question) > 0
    
def test_vizgen():
    summary = sandesh.summarize(
        cars_data_url,
        summary_method="llmt")

    goals = sandesh.goals(summary,llm,n=2)
    charts = sandesh.visualize(
        summary=summary,
        goal=goals[0],
        llm=llm,
        library="seaborn")

    assert len(charts) > 0
    first_chart = charts[0]

    # Ensure the first chart has a status of True
    assert first_chart.status is True

    # Ensure no errors in the first chart
    assert first_chart.error is None

    # Ensure the raster image of the first chart exists
    assert len(first_chart.raster) > 0

    # Test saving the raster image of the first chart
    temp_file_path = "temp_image.png"
    first_chart.savefig(temp_file_path)
    # Ensure the image is saved correctly
    assert os.path.exists(temp_file_path)
    # Clean up
    os.remove(temp_file_path)
    
test_vizgen()





