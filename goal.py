import json
from utils import clean_code_snippet
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
# from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

SYSTEM_INSTRUCTIONS = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""







# model_type = "azureOAi"
# llm = HF_endpoint(model_type)


class Goal:
    """A visualization goal"""
    def __init__(self, question: str, visualization: str, rationale: str, index: Optional[int] = 0):
        self.question = question
        self.visualization = visualization
        self.rationale = rationale
        self.index = index

    def _repr_markdown_(self):
        return f"""
### Goal {self.index}
---
**Question:** {self.question}

**Visualization:** `{self.visualization}`

**Rationale:** {self.rationale}
"""


class Persona:
    """A persona"""
    persona: str
    rationale: str

    def _repr_markdown_(self):
        return f"""
### Persona
---

**Persona:** {self.persona}

**Rationale:** {self.rationale}
"""


class GoalExplorer():
    """Generate goals given a summary of data"""
    def __init__(self) -> None:
        pass
    
    def generate(self,summary: dict,llm,n=5,persona: Persona=None) -> list[Goal] :
        """Generate goals given a summary of data"""
        
        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {summary} \n\n"""
        
        if not persona:
            persona = Persona()
            persona.persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
            persona.rationale=""

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona} persona, who is insterested in complex, insightful goals about the data. \n"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant",
             "content":
             f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n "}]
        response = llm.invoke(messages)
        result: list[Goal] = response.content
        print("result:-"+"\n")
        print(result)
        try:
            #json_string = clean_code_snippet(result)
            json_string = result
            if isinstance(json_string, str):
                json_string =json_string.replace('\n', '').replace('\\n', '')
                result = json.loads(json_string)
            else :
                result = json_string
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Goal(**x) for x in result]
        except json.decoder.JSONDecodeError:
           # logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate goals. Consider using a larger model or a model with higher max token length.")
        return result   
    
    
    
# summary = {'name': 'cars.csv', 'file_name': 'cars.csv', 'dataset_description': 'This dataset contains information about cars.', 'fields': [{'column': 'Name', 'properties': {'dtype': 'string', 'samples': ['Nissan Altima S 4dr', 'Mercury Marauder 4dr', 'Toyota Prius 4dr (gas/electric)'], 'num_unique_values': 385, 'semantic_type': 'car_model', 'description': 'The name of the car model.'}}, {'column': 'Type', 'properties': {'dtype': 'category', 'samples': ['SUV', 'Minivan', 'Sports Car'], 'num_unique_values': 5, 'semantic_type': 'car_type', 'description': 'The type of the car.'}}, {'column': 'AWD', 'properties': {'dtype': 'int64', 'samples': [1, 0], 'num_unique_values': 2, 'semantic_type': 'boolean', 'description': 'Whether the car has all-wheel drive (1) or not (0).'}}, {'column': 'RWD', 'properties': {'dtype': 'int64', 'samples': [1, 0], 'num_unique_values': 2, 'semantic_type': 'boolean', 'description': 'Whether the car has rear-wheel drive (1) or not (0).'}}, {'column': 'Retail_Price', 'properties': {'dtype': 'int64', 'samples': [22775, 37245], 'num_unique_values': 370, 'semantic_type': 'price', 'description': 'The retail price of the car.'}}, {'column': 'Dealer_Cost', 'properties': {'dtype': 'int64', 'samples': [18030, 31558], 'num_unique_values': 384, 'semantic_type': 'price', 'description': 'The cost of the car for the dealer.'}}, {'column': 'Engine_Size__l_', 'properties': {'dtype': 'number', 'std': 1.0266787710109588, 'min': 0.0, 'max': 6.0, 'samples': [2.2, 5.3], 'num_unique_values': 40, 'semantic_type': 'engine_size', 'description': "The size of the car's engine in liters."}}, {'column': 'Cyl', 'properties': {'dtype': 'int64', 'samples': [4, 9], 'num_unique_values': 8, 'semantic_type': 'number_of_cylinders', 'description': "The number of cylinders in the car's engine."}}, {'column': 'Horsepower_HP_', 'properties': {'dtype': 'int64', 'samples': [126, 138], 'num_unique_values': 100, 'semantic_type': 'horsepower', 'description': "The horsepower of the car's engine."}}, {'column': 'City_Miles_Per_Gallon', 'properties': {'dtype': 'int64', 'samples': [59, 32], 'num_unique_values': 29, 'semantic_type': 'fuel_efficiency', 'description': 'The miles per gallon the car can achieve in city driving.'}}, {'column': 'Highway_Miles_Per_Gallon', 'properties': {'dtype': 'int64', 'samples': [43, 37], 'num_unique_values': 33, 'semantic_type': 'fuel_efficiency', 'description': 'The miles per gallon the car can achieve on the highway.'}}, {'column': 'Weight', 'properties': {'dtype': 'int64', 'samples': [4473, 3472], 'num_unique_values': 315, 'semantic_type': 'weight', 'description': 'The weight of the car in pounds.'}}, {'column': 'Wheel_Base', 'properties': {'dtype': 'int64', 'samples': [110, 0], 'num_unique_values': 34, 'semantic_type': 'length', 'description': 'The wheelbase of the car in inches.'}}, {'column': 'Len', 'properties': {'dtype': 'int64', 'samples': [197, 183], 'num_unique_values': 60, 'semantic_type': 'length', 'description': 'The length of the car in inches.'}}, {'column': 'Width', 'properties': {'dtype': 'int64', 'samples': [72, 70], 'num_unique_values': 19, 'semantic_type': 'length', 'description': 'The width of the car in inches.'}}], 'field_names': ['Name', 'Type', 'AWD', 'RWD', 'Retail_Price', 'Dealer_Cost', 'Engine_Size__l_', 'Cyl', 'Horsepower_HP_', 'City_Miles_Per_Gallon', 'Highway_Miles_Per_Gallon', 'Weight', 'Wheel_Base', 'Len', 'Width']}

# P = GoalExplorer()
# result = P.generate(summary,llm)
# print(result)        
         