from scaffold import ChartScaffold
import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
from llm_config import HF_endpoint
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI
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

# class Summary:
#     """A summary of a dataset"""

#     name: str
#     file_name: str
#     dataset_description: str
#     field_names: List[Any]
#     fields: Optional[List[Any]] = None

#     def _repr_markdown_(self):
#         field_lines = "\n".join([f"- **{name}:** {field}" for name,
#                                 field in zip(self.field_names, self.fields)])
#         return f""
    
system_prompt = """
You are a high skilled visualization assistant that can modify a provided visualization code based on a set of instructions. You MUST return a full program. DO NOT include any preamble text. Do not include explanations or prose.
"""


class VizEditor():
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, summary, instructions: list[str],llm,
             library='altair'):
        """Edit a code spec based on instructions"""

        instruction_string = ""
        for i, instruction in enumerate(instructions):
            instruction_string += f"{i+1}. {instruction} \n"

        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)
        # print("instructions", instructions)

        messages = [
            {
                "role": "system", "content": system_prompt}, {
                "role": "system", "content": f"The dataset summary is : \n\n {summary} \n\n"}, {
                "role": "system", "content": f"The modifications you make MUST BE CORRECT and  based on the '{library}' library and also follow these instructions \n\n{library_instructions} \n\n. The resulting code MUST use the following template \n\n {library_template} \n\n "}, {
                    "role": "user", "content": f"ALL ADDITIONAL LIBRARIES USED MUST BE IMPORTED.\n The code to be modified is: \n\n{code} \n\n. YOU MUST THINK STEP BY STEP, AND CAREFULLY MODIFY ONLY the content of the plot(..) method TO MEET EACH OF THE FOLLOWING INSTRUCTIONS: \n\n {instruction_string} \n\n. The completed modified code THAT FOLLOWS THE TEMPLATE above is. \n"}]
        response = llm.invoke(messages)
        completions = response.content
        return [x['content'] for x in completions]
