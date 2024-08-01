from dataclasses import asdict
from typing import Dict
from scaffold import ChartScaffold
import json
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
from typing import Any, Dict, List, Optional, Union
from utils import clean_code_snippet
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

system_prompt = """
You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some code template, you complete the template to generate a visualization given the dataset and the goal described. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc. .
"""


class VizGenerator():
    """Generate visualizations from prompt"""

    def __init__(
        self
    ) -> None:

        self.scaffold = ChartScaffold()

    def generate(self, summary: Dict, goal: Goal,
                 llm, library='altair'):
        """Generate visualization code given a summary and a goal"""

        library_template, library_instructions = self.scaffold.get_template(goal, library)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary} \n\n"},
            library_instructions,
            {"role": "user",
             "content":
             f"The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL PYTHON PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation. \n\n THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {library_template} \n\n.The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n"}]

        response = llm.invoke(messages)
        completions = response.content
        #print(completions)
        #print(type(completions))
        #completions = clean_code_snippet(completions)
        #response = [x['content'] for x in completions]
        response = completions

        print("vizgenearotr result:-"+"\n")
        #print(response)

        return [response]
    
# sum = {'name': 'cars.csv', 'file_name': 'cars.csv', 'dataset_description': 'This dataset contains information about cars.', 'fields': [{'column': 'Name', 'properties': {'dtype': 'string', 'samples': ['Nissan Altima S 4dr', 'Mercury Marauder 4dr', 'Toyota Prius 4dr (gas/electric)'], 'num_unique_values': 385, 'semantic_type': 'car_model', 'description': 'The name of the car model.'}}, {'column': 'Type', 'properties': {'dtype': 'category', 'samples': ['SUV', 'Minivan', 'Sports Car'], 'num_unique_values': 5, 'semantic_type': 'car_type', 'description': 'The type of the car.'}}, {'column': 'AWD', 'properties': {'dtype': 'int64', 'samples': [1, 0], 'num_unique_values': 2, 'semantic_type': 'boolean', 'description': 'Whether the car has all-wheel drive (1) or not (0).'}}, {'column': 'RWD', 'properties': {'dtype': 'int64', 'samples': [1, 0], 'num_unique_values': 2, 'semantic_type': 'boolean', 'description': 'Whether the car has rear-wheel drive (1) or not (0).'}}, {'column': 'Retail_Price', 'properties': {'dtype': 'int64', 'samples': [22775, 37245], 'num_unique_values': 370, 'semantic_type': 'price', 'description': 'The retail price of the car.'}}, {'column': 'Dealer_Cost', 'properties': {'dtype': 'int64', 'samples': [18030, 31558], 'num_unique_values': 384, 'semantic_type': 'price', 'description': 'The cost of the car for the dealer.'}}, {'column': 'Engine_Size__l_', 'properties': {'dtype': 'number', 'std': 1.0266787710109588, 'min': 0.0, 'max': 6.0, 'samples': [2.2, 5.3], 'num_unique_values': 40, 'semantic_type': 'engine_size', 'description': "The size of the car's engine in liters."}}, {'column': 'Cyl', 'properties': {'dtype': 'int64', 'samples': [4, 9], 'num_unique_values': 8, 'semantic_type': 'number_of_cylinders', 'description': "The number of cylinders in the car's engine."}}, {'column': 'Horsepower_HP_', 'properties': {'dtype': 'int64', 'samples': [126, 138], 'num_unique_values': 100, 'semantic_type': 'horsepower', 'description': "The horsepower of the car's engine."}}, {'column': 'City_Miles_Per_Gallon', 'properties': {'dtype': 'int64', 'samples': [59, 32], 'num_unique_values': 29, 'semantic_type': 'fuel_efficiency', 'description': 'The miles per gallon the car can achieve in city driving.'}}, {'column': 'Highway_Miles_Per_Gallon', 'properties': {'dtype': 'int64', 'samples': [43, 37], 'num_unique_values': 33, 'semantic_type': 'fuel_efficiency', 'description': 'The miles per gallon the car can achieve on the highway.'}}, {'column': 'Weight', 'properties': {'dtype': 'int64', 'samples': [4473, 3472], 'num_unique_values': 315, 'semantic_type': 'weight', 'description': 'The weight of the car in pounds.'}}, {'column': 'Wheel_Base', 'properties': {'dtype': 'int64', 'samples': [110, 0], 'num_unique_values': 34, 'semantic_type': 'length', 'description': 'The wheelbase of the car in inches.'}}, {'column': 'Len', 'properties': {'dtype': 'int64', 'samples': [197, 183], 'num_unique_values': 60, 'semantic_type': 'length', 'description': 'The length of the car in inches.'}}, {'column': 'Width', 'properties': {'dtype': 'int64', 'samples': [72, 70], 'num_unique_values': 19, 'semantic_type': 'length', 'description': 'The width of the car in inches.'}}], 'field_names': ['Name', 'Type', 'AWD', 'RWD', 'Retail_Price', 'Dealer_Cost', 'Engine_Size__l_', 'Cyl', 'Horsepower_HP_', 'City_Miles_Per_Gallon', 'Highway_Miles_Per_Gallon', 'Weight', 'Wheel_Base', 'Len', 'Width']}
# goals = [
#     {
#         "index": 0,
#         "question": "What is the distribution of car types in the dataset?",
#         "visualization": "bar chart of car types",
#         "rationale": "By visualizing the distribution of car types, we can understand the proportion of each car type in the dataset. This information can be useful for market analysis and understanding consumer preferences."
#     },
#     {
#         "index": 1,
#         "question": "What is the relationship between engine size and horsepower?",
#         "visualization": "scatter plot of engine size vs horsepower",
#         "rationale": "By plotting engine size against horsepower, we can identify any patterns or correlations between these two variables. This can help us understand the performance characteristics of different car models."
#     },
#     {
#         "index": 2,
#         "question": "What is the average retail price of cars with all-wheel drive (AWD) vs rear-wheel drive (RWD)?",
#         "visualization": "bar chart of average retail price by drive type",
#         "rationale": "Comparing the average retail price of cars with AWD and RWD can provide insights into the pricing differences between these two types of drivetrains. This information can be valuable for pricing strategies and market positioning."
#     },
#     {
#         "index": 3,
#         "question": "What is the distribution of fuel efficiency (city miles per gallon) for different car types?",
#         "visualization": "box plot of fuel efficiency by car type",
#         "rationale": "A box plot can show the distribution of fuel efficiency for different car types, including the median, quartiles, and any outliers. This visualization can help us compare the fuel efficiency of different car types and identify any variations or trends."
#     },
#     {
#         "index": 4,
#         "question": "What is the relationship between car weight and fuel efficiency (highway miles per gallon)?",
#         "visualization": "scatter plot of car weight vs highway miles per gallon",
#         "rationale": "By plotting car weight against highway miles per gallon, we can analyze the impact of weight on fuel efficiency. This can help us understand the trade-off between car weight and fuel consumption, and potentially identify opportunities for improving fuel efficiency."
#     }
# ]



# p = VizGenerator()
# res = p.generate(sum,goals,llm)
# print(res)