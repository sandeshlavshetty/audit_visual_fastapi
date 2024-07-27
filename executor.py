# from dataclasses import dataclass
import base64
from dataclasses import field
from typing import Any, Dict, List, Optional, Union
from pydantic.dataclasses import dataclass
import ast
import importlib
import io
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import warnings
import altair as alt
import base64
from io import BytesIO

class Summary:
    """A summary of a dataset"""

    def __init__(self, name: str, file_name: str, dataset_description: str, field_names: List[Any], fields: Optional[List[Any]] = None):
        self.name = name
        self.file_name = file_name
        self.dataset_description = dataset_description
        self.field_names = field_names
        self.fields = fields

    def _repr_markdown_(self):
        field_lines = "\n".join([f"- **{name}:** {field}" for name,
                                 field in zip(self.field_names, self.fields)])
        return f"""
## Dataset Summary

---

**Name:** {self.name}

**File Name:** {self.file_name}

**Dataset Description:**

{self.dataset_description}

**Fields:**

{field_lines}
"""


class ChartExecutorResponse:
    """Response from a visualization execution"""

    def __init__(self, spec: Optional[Union[str, Dict]], status: bool, raster: Optional[str], code: str, library: str, error: Optional[Dict] = None):
        self.spec = spec
        self.status = status
        self.raster = raster
        self.code = code
        self.library = library
        self.error = error

    def _repr_mimebundle_(self, include=None, exclude=None):
        print("repr bundle function starts")
        bundle = {"text/plain": self.code}
        if self.raster is not None:
            print("self.raster if started")
            bundle["image/png"] = self.raster
            print("self.rasted if done")
        if self.spec is not None:
            bundle["application/vnd.vegalite.v5+json"] = self.spec
            print("self.spec if")

        return bundle

    def savefig(self, path):
        """Save the raster image to a specified path if it exists"""
        if self.raster:
            with open(path, 'wb') as f:
                f.write(base64.b64decode(self.raster))
        else:
            raise FileNotFoundError("No raster image to save")
    
    def encode_img(self):
        return self.raster
        

def preprocess_code(code: str) -> str:
    """Preprocess code to remove any preamble and explanation text"""
    print("strated procescode function")
    code = code.replace("<imports>", "")
    code = code.replace("<stub>", "")
    code = code.replace("<transforms>", "")

    # remove all text after chart = plot(data)
    if "chart = plot(data)" in code:
        # print(code)
        index = code.find("chart = plot(data)")
        if index != -1:
            code = code[: index + len("chart = plot(data)")]

    if "```" in code:
        pattern = r"```(?:\w+\n)?([\s\S]+?)```"
        matches = re.findall(pattern, code)
        if matches:
            code = matches[0]
        # code = code.replace("```", "")
        # return code

    if "import" in code:
        # return only text after the first import statement
        index = code.find("import")
        if index != -1:
            code = code[index:]

    code = code.replace("```", "")
    if "chart = plot(data)" not in code:
        code = code + "\nchart = plot(data)"
    print("ended process code function and reuslt it returns is :-"+"\n")
    print(code)
    return code


def get_globals_dict(code_string, data):
    # Parse the code string into an AST
    print("entered get globals function")
    tree = ast.parse(code_string)
    # Extract the names of the imported modules and their aliases
    imported_modules = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = importlib.import_module(alias.name)
                imported_modules.append((alias.name, alias.asname, module))
        elif isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(module, alias.name)
                imported_modules.append(
                    (f"{node.module}.{alias.name}", alias.asname, obj)
                )

    # Import the required modules into a dictionary
    globals_dict = {}
    for module_name, alias, obj in imported_modules:
        if alias:
            globals_dict[alias] = obj
        else:
            globals_dict[module_name.split(".")[-1]] = obj

    ex_dicts = {"pd": pd, "data": data, "plt": plt}
    globals_dict.update(ex_dicts)
    print("get globals function done:-"+"\n")
    print(globals_dict)
    return globals_dict



class ChartExecutor:
    """Execute code and return chart object"""

    def __init__(self) -> None:
        pass

    def execute(
        self,
        code_specs: List[str],
        data: Any,
        summary: Summary,
        library="altair",
        return_error: bool = False,
    ) -> Any:
        """Validate and convert code"""

        # # check if user has given permission to execute code. if env variable
        # # LIDA_ALLOW_CODE_EVAL is set to '1'. Else raise exception
        # if os.environ.get("LIDA_ALLOW_CODE_EVAL") != '1':
        #     raise Exception(
        #         "Permission to execute code not granted. Please set the environment variable LIDA_ALLOW_CODE_EVAL to '1' to allow code execution.")
        print("execute function started")
        if isinstance(summary, dict): 
            summary = Summary(**summary)
        print("if statement passed")
        charts = []
        code_spec_copy = code_specs.copy()
        print("processcode func called")
        code_specs = [preprocess_code(code) for code in code_specs]
        if library == "altair":
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]
                    vega_spec = chart.to_dict()
                    del vega_spec["data"]
                    if "datasets" in vega_spec:
                        del vega_spec["datasets"]

                    vega_spec["data"] = {"url": f"/files/data/{summary.file_name}"}
                    charts.append(
                        ChartExecutorResponse(
                            spec=vega_spec,
                            status=True,
                            raster=None,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code_spec_copy, "\n===========\n")
                    print(exception_error)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts
        elif library == "matplotlib" or library == "seaborn":
            # print colum dtypes
            for code in code_specs:
                try:
            # Suppress specific warnings
                    ex_locals = get_globals_dict(code, data)
                    print("ex_locals result:-"+"\n")
                    print(ex_locals)
                    print(exec(code, ex_locals))
                    chart = ex_locals["chart"]
                    print("chart print:-"+"\n")
                    print(chart)
                    if plt:
                        print("entered if plt")
                        buf = io.BytesIO()
                        plt.box(False)
                        plt.grid(color="lightgray", linestyle="dashed", zorder=-10)
                        # try:
                        #     plt.draw()
                        #     # plt.tight_layout()
                        # except AttributeError:
                        #     print("Warning: tight_laclearyout encountered an error. The layout may not be optimal.")
                        #     pass

                        plt.savefig(buf, format="png", dpi=100, pad_inches=0.2)
                        buf.seek(0)
                        plot_data = base64.b64encode(buf.read()).decode("ascii")
                        plt.close()
                    print("\n"+"charsts.append function called")
                    charts.append(
                        ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code_spec_copy[0])
                    print("****\n", str(exception_error))
                    # print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts
        elif library == "ggplot":
            # print colum dtypes
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]
                    if plt:
                        buf = io.BytesIO()
                        chart.save(buf, format="png")
                        plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    charts.append(
                        ChartExecutorResponse(
                            spec=None,
                            status=True,
                            raster=plot_data,
                            code=code,
                            library=library,
                        )
                    )
                except Exception as exception_error:
                    print(code)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts

        elif library == "plotly":
            for code in code_specs:
                try:
                    ex_locals = get_globals_dict(code, data)
                    exec(code, ex_locals)
                    chart = ex_locals["chart"]

                    if pio:
                        chart_bytes = pio.to_image(chart, 'png')
                        plot_data = base64.b64encode(chart_bytes).decode('utf-8')

                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=True,
                                raster=plot_data,
                                code=code,
                                library=library,
                            )
                        )
                except Exception as exception_error:
                    print(code)
                    print(traceback.format_exc())
                    if return_error:
                        charts.append(
                            ChartExecutorResponse(
                                spec=None,
                                status=False,
                                raster=None,
                                code=code,
                                library=library,
                                error={
                                    "message": str(exception_error),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                        )
            return charts

        else:
            raise Exception(
                f"Unsupported library. Supported libraries are altair, matplotlib, seaborn, ggplot, plotly. You provided {library}"
            )
            
            