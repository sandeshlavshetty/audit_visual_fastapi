import os
from typing import List, Union
import pandas as pd
from utils import read_dataframe,get_data
from summarizer import Summarizer
from goal import GoalExplorer
from persona import PersonaExplorer
from executor import ChartExecutor
from viz import VizGenerator, VizEditor, VizExplainer, VizEvaluator, VizRepairer, VizRecommender,  VizInference
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
# from dataclasses import dataclass
import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
#cars_data_url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"

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

# model_type = "azureOAi"
# llm = HF_endpoint(model_type)


class Manager():
    def __init__(self) -> None:
        self.summarizer = Summarizer()
        self.summarizer = Summarizer()
        self.goal = GoalExplorer()
        self.vizgen = VizGenerator()
        self.vizeditor = VizEditor()
        self.executor = ChartExecutor()
        self.explainer = VizExplainer()
        self.evaluator = VizEvaluator()
        self.repairer = VizRepairer()
        self.recommender = VizRecommender()
        self.data = None
        self.infographer = None
        self.persona = PersonaExplorer()         
        self.inference = VizInference()     
         
    def summarize(self,
        data: Union[pd.DataFrame, str],
        file_name="",
        n_samples: int = 3,
        summary_method: str = "default"):

        if isinstance(data, str):
            # file_name = data.split("/")[-1]
            data = get_data(data)
            
        self.data = data
        return self.summarizer.summarize(
            data=self.data,file_name=file_name, n_samples=n_samples,
            summary_method=summary_method)
        
    def goals(
        self,
        summary,
        llm,
        n: int = 5,
        persona: Persona = None
    ) -> List[Goal]:
        """
        Generate goals based on a summary.

        Args:
            summary (Summary): Input summary.
            textgen_config (TextGenerationConfig, optional): Text generation configuration. Defaults to TextGenerationConfig().
            n (int, optional): Number of goals to generate. Defaults to 5.
            persona (Persona, str, dict, optional): Persona information. Defaults to None.

        Returns:
            List[Goal]: List of generated goals.

        Example of list of goals:

            Goal 0
            Question: What is the distribution of Retail_Price?

            Visualization: histogram of Retail_Price

            Rationale: This tells about the spread of prices of cars in the dataset.

            Goal 1
            Question: What is the distribution of Horsepower_HP_?

            Visualization: box plot of Horsepower_HP_

            Rationale: This tells about the distribution of horsepower of cars in the dataset.
        """
        # self.check_textgen(config=textgen_config)

        if isinstance(persona, dict):
            persona = Persona(**persona)
        if isinstance(persona, str):
            persona = Persona(persona=persona, rationale="")

        return self.goal.generate(summary=summary, llm=llm, n=n, persona=persona)     


    def personas(
            self, summary, llm,
            n=5):
        #self.check_textgen(config=textgen_config)

        return self.persona.generate(summary=summary, llm=llm, n=n)

 
    def visualize(
        self,
        summary,
        goal,
        llm,
        library="seaborn",
        return_error: bool = False,
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        #self.check_textgen(config=textgen_config)
        code_specs = self.vizgen.generate(
            summary=summary, goal=goal, llm=llm,
            library=library)
        print("code_specs result:-"+"\n")
        print(code_specs)
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts
 
 
    def visualize_all(
        self,
        summary,
        goal,
        llm,
        library="seaborn",
        return_error: bool = False,
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str):
            goal = Goal(question=goal, visualization=goal, rationale="")

        #self.check_textgen(config=textgen_config)
        code_specs = self.vizgen.generate(
            summary=summary, goal=goal, llm=llm,
            library=library)
        print("code_specs result:-"+"\n")
        print(code_specs)
        vizevalfeed = self.evaluator.generate(
        code_specs,goal,
        llm=llm,
        library=library
    )    
        print("viz feedback results"+"\n")
        print(vizevalfeed)
        vizreapaircode = self.repairer.generate(
            code_specs,vizevalfeed,goal,summary=summary,llm=llm,library=library
        )
        print("viz repair done")
        charts = self.execute(
            code_specs=vizreapaircode,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        print("charts got created")
        return charts
 


    def inference_gen(self,image_b64,llm):
        print("manager inference func entered")
        return self.inference.genrate(image_b64,llm)

    def execute(
        self,
        code_specs,
        data,
        summary,
        library: str = "seaborn",
        return_error: bool = False,
    ):

        # if data is None:      # note understood this part
        #     root_file_path = os.path.dirname(os.path.abspath(lida.__file__))
        #     print(root_file_path)
        #     data = read_dataframe(
        #         os.path.join(root_file_path, "files/data", summary.file_name)
        #     )

        # col_properties = summary.properties

        return self.executor.execute(
            code_specs=code_specs,
            data=data,
            summary=summary,
            library=library,
            return_error=return_error,
        )



    def edit(
        self,
        code,
        summary,
        instructions: List[str],
        llm,
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        #self.check_textgen(config=textgen_config)

        if isinstance(instructions, str):
            instructions = [instructions]

        code_specs = self.vizeditor.generate(
            code=code,
            summary=summary,
            instructions=instructions,
            llm=llm,
            library=library,
        )

        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts
    
    def repair(
        self,
        code,
        goal: Goal,
        summary,
        feedback,
        llm,
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """ Repair a visulization given some feedback"""
        #self.check_textgen(config=textgen_config)
        code_specs = self.repairer.generate(
            code=code,
            feedback=feedback,
            goal=goal,
            summary=summary,
            llm=llm,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts


    def explain(
        self,
        code,
        llm,
        library: str = "seaborn",
    ):
        """Explain a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """
        #self.check_textgen(config=textgen_config)
        return self.explainer.generate(
            code=code,
            llm=llm,
            library=library,
        )
        
        
    def evaluate(
        self,
        code,
        goal: Goal,
        llm,
        library: str = "seaborn",
    ):
        """Evaluate a visualization code given a goal

        Args:
            code (_type_): _description_
            goal (Goal): A visualization goal

        Returns:
            _type_: _description_
        """

        #self.check_textgen(config=textgen_config)

        return self.evaluator.generate(
            code=code,
            goal=goal,
            llm=llm,
            library=library,
        )


    def recommend(
        self,
        code,
        summary,
        llm,
        n=4,
        library: str = "seaborn",
        return_error: bool = False,
    ):
        """Edit a visualization code given a set of instructions

        Args:
            code (_type_): _description_
            instructions (List[Dict]): A list of instructions

        Returns:
            _type_: _description_
        """

        #self.check_textgen(config=textgen_config)

        code_specs = self.recommender.generate(
            code=code,
            summary=summary,
            llm = llm,            
            n=n,
            library=library,
        )
        charts = self.execute(
            code_specs=code_specs,
            data=self.data,
            summary=summary,
            library=library,
            return_error=return_error,
        )
        return charts

# P = Manager()
# result = P.summarize(cars_data_url,summary_method="llmt")
# print(result)        


         