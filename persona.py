import json
import logging
from lida.datamodel import Persona, TextGenerationConfig

class Persona:
    """A persona"""
    def __init__(self, persona: str, rationale: str) -> None:
        self.persona = persona
        self.rationale = rationale

    def _repr_markdown_(self) -> str:
        return f"""
### Persona
---

**Persona:** {self.persona}

**Rationale:** {self.rationale}
"""


system_prompt = """You are an experienced data analyst  who can take a dataset summary and generate a list of n personas (e.g., ceo or accountant for finance related data, economist for population or gdp related data, doctors for health data, or just users) that might be critical stakeholders in exploring some data and describe rationale for why they are critical. The personas should be prioritized based on their relevance to the data. Think step by step.

Your response should be perfect JSON in the following format:
```[{"persona": "persona1", "rationale": "..."},{"persona": "persona1", "rationale": "..."}]```
"""



class PersonaExplorer():
    """Generate personas given a summary of data"""

    def __init__(self) -> None:
        pass

    def generate(self, summary: dict,llm,n=5) -> list[Persona]:
        """Generate personas given a summary of data"""

        user_prompt = f"""The number of PERSONAs to generate is {n}. Generate {n} personas in the right format given the data summary below,\n .
        {summary} \n""" + """

        .
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": user_prompt},
        ]
        response = llm.invoke(messages)
        result = response.content
        print("persona.py result :- "+"\n")
        #print(result)
        try:
            json_string = result
            if isinstance(json_string, str):
                json_string =json_string.replace('\n', '').replace('\\n', '')
                result = json.loads(json_string)
            else :
                result = json_string            
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Persona(**x) for x in result]
        except json.decoder.JSONDecodeError:
            #logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate personas.  Consider using a larger model or a model with higher max token length.")
        return result