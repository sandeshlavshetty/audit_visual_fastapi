
import json
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

You are a helpful assistant highly skilled in recommending a DIVERSE set of visualization code. Your input is an example visualization code,  a summary of a dataset and an example visualization goal that the user has already seen. Given this input, your task is to recommend additional visualizations that a user may be interested. Your recommendation may consider different types of valid data aggregations, chart types, clearer ways of displaying information and uses different variables from the data summary. THE CODE YOU GENERATE MUST BE CORRECT (follow the language syntax and syntax of the visualization grammar) AND FOLLOW VISUALIZATION BEST PRACTICES.

Your output MUST be a n code snippets separated by ******* (5 asterisks). Each snippet MUST BE AN independent code snippet (with one plot method) similar to the example code. For example

```python
# code snippet 1
import  ...
....
```
*****

```python
# code snippet 2
import ...
....
```

```python
# code snippet n
import ...
....
```


"""




class VizRecommender():
    """Generate visualizations from prompt"""

    def __init__(
        self,
    ) -> None:
        self.scaffold = ChartScaffold()

    def generate(
            self, code: str, summary,
            llm,
            n=3,
            library='seaborn'):
        """Recommend a code spec based on existing visualization"""

        library_template, library_instructions = self.scaffold.get_template(Goal(
            index=0,
            question="",
            visualization="",
            rationale=""), library)

        structure_instruction = f"""
        EACH CODE SNIPPET MUST BE A FULL PROGRAM (IT MUST IMPORT ALL THE LIBRARIES THAT ARE USED AND MUST CONTAIN plot(data) method). IT MUST FOLLOW THE STRUCTURE BELOW AND ONLY MODIFY THE INDICATED SECTIONS. \n\n {library_template} \n\n.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": structure_instruction},
            {"role": "system", "content": f"The dataset summary is : \n\n {summary} \n\n"},
            {"role": "system",
             "content":
             f"An example visualization code is: \n\n ```{code}``` \n\n. You MUST use only the {library} library. \n"},
            {"role": "user", "content": f"Recommend {n} (n=({n})) visualizations in the format specified. \n."}]

        response = llm.invoke(messages)
        result = response.content
        output = []
        #snippets = result.text[0]["content"].split("*****")
        snippets = result
        for snippet in snippets:
            if isinstance(snippet,str):
                cleaned_snippet = clean_code_snippet(snippet)
            else :
                 cleaned_snippet = snippet   
            if len(cleaned_snippet) > 4:
                output.append(cleaned_snippet)

        return output