from typing import Dict, List, Union

from scaffold import ChartScaffold
import json
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
from utils import clean_code_snippet
from typing import Any, Dict, List, Optional, Union


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
You are a helpful assistant highly skilled in revising visualization code to improve the quality of the code and visualization based on feedback.  Assume that data in plot(data) contains a valid dataframe.
You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizRepairer():
    """Fix visualization code based on feedback"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, feedback: Union[str, Dict, List[Dict]],
            goal: Goal, summary, llm, library='altair',):
        """Fix a code spec based on feedback"""
        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"The dataset summary is : {summary}. \n . The original goal was: {goal}."},
            {"role": "system",
             "content":
             f"You MUST use only the {library}. The resulting code MUST use the following template {library_template}. Only use variables that have been defined in the code or are in the dataset summary"},
            {"role": "user", "content": f"The existing code to be fixed is: {code}. \n Fix the code above to address the feedback: {feedback}. ONLY apply feedback that are CORRECT."}]

        # library with the following instructions {library_instructions}

        response = llm.invoke(messages)
        completions = response.content
        return [completions]