import json
from typing import Union
import pandas as pd
from llm_config import HF_endpoint
from utils import clean_code_snippet,read_dataframe
import warnings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
# result = llm.invoke("tell me a joke")
# print(result)

system_prompt = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You must return an updated JSON dictionary without any preamble or explanation.
"""

class Summarizer():
    def __init__(self) -> None:
        self.summary = None

  
    
    def check_type(self, dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                # Check if the string column can be cast to a valid datetime
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "date"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            # add min max if dtype is date
            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            # Add additional properties to the output dictionary
            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(
                    n_samples, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            properties_list.append(
                {"column": column, "properties": properties})

        return properties_list

    def enrich(self, base_summary: dict,llm,model_type) -> dict:
        """Enrich the data summary with descriptions"""
        #.info(f"Enriching the data summary with descriptions")
        if model_type == "hf" :
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                content=f"""
                Annotate the dictionary below. Only return a JSON object with proper {{ and }} closing.
                {base_summary}
                """
                ),
                ]
            print("Enrich function working")
            chat_model = ChatHuggingFace(llm=llm)
            res = chat_model.invoke(messages)
            response = res.content  #call 1 
            #print(response)
            enriched_summary = base_summary
            try:
                #json_string = clean_code_snippet(response.text[0]["content"])
                json_string = clean_code_snippet(response)
                #print(json_string)
                enriched_summary = json.loads(json_string)
            except json.decoder.JSONDecodeError:
                error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a default summary or  a larger model with higher max token length. | {response.text[0]['content']}"
                #logger.info(error_msg)
                #print(response.text[0]["content"])
                #print(response)
                raise ValueError(error_msg + "" + response.usage)
            return enriched_summary
        elif model_type == "azureOAi" :
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": f"""
                Annotate the dictionary below. Only return a JSON object.
                {base_summary}
                """},
                ]
            print("Enrich function working with azure OAi")
            res = llm.invoke(messages)
            response = res.content  #call 1 
            #print(response)
            enriched_summary = base_summary
            try:
                #json_string = clean_code_snippet(response.text[0]["content"])
                #json_string = clean_code_snippet(response)
                json_string = response
                print("json_string:-"+"\n")
                cleaned_json_content = json_string.replace('\n', '').replace('\\n', '')
                cleaned_json = json.loads(cleaned_json_content) # no need to load text to json if it returns json data
                enriched_summary = cleaned_json
            except json.decoder.JSONDecodeError:            
                error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a default summary or  a larger model with higher max token length. | {response.text[0]['content']}"
                #logger.info(error_msg)
                #print(response.text[0]["content"])
                #print(response)
                raise ValueError(error_msg + "" + response.usage)
            return enriched_summary        
        else :
            print("Chat model not configued")

    def summarize(self, data: Union[pd.DataFrame, str],file_name="", n_samples: int = 3,
                  summary_method: str = "default", 
                  encoding: str = 'utf-8') -> dict:
        """Summarize data from a pandas DataFrame or a file location"""

        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            # modified to include encoding
            data = read_dataframe(data, encoding=encoding)
        data_properties = self.get_column_properties(data, n_samples)
        print("data properties:-")
        #print(data_properties)
        # default single stage summary construction
        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }

        data_summary = base_summary
        model_type = "azureOAi"
        llm = HF_endpoint(model_type)
        if summary_method == "llmt":
            # two stage summarization with llm enrichment
            data_summary = self.enrich(
                base_summary,
                llm,model_type)
        elif summary_method == "columns":
            # no enrichment, only column names
            data_summary = {
                "name": file_name,
                "file_name": file_name,
                "dataset_description": ""
            }

        data_summary["field_names"] = data.columns.tolist()
        data_summary["file_name"] = file_name

        return data_summary


#note :- to create summary two methods :- 1) by llm set summary_method = llmt 2) for just columns summary_method= columns


#cars_data_url = "https://raw.githubusercontent.com/uwdata/draco/master/data/movies.csv"
#cars_data_url = "https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data"
# cars_data_url ="movies.json"
# #cars_data_url =https://bob-hack.vercel.app/file-viewer/movies.csv


# P = Summarizer()
# result = P.summarize(cars_data_url,summary_method="llmt")
# print(result) 