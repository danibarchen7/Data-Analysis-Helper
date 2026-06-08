import os
import json
import io
import base64
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- Schemas ---
ANALYSIS_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "investigation_goals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "hypotheses": {"type": "ARRAY", "items": {"type": "STRING"}},
        "planned_steps": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["investigation_goals", "hypotheses", "planned_steps"]
}

FINAL_INSIGHTS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary_of_findings": {"type": "STRING"},
        "key_metrics": {"type": "OBJECT"},
        "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["summary_of_findings", "key_metrics", "recommendations"]
}

class GeminiDataAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analyze_bytes_data(self, df: pd.DataFrame, user_request: str):
        data_schema = f"Columns: {list(df.columns)}\nData Types:\n{df.dtypes.to_string()}\nShape: {df.shape}"
        csv_content = df.to_csv(index=False)

        # --- STEP 1: PLANNING ---
        planning_prompt = f"""
        You are an expert Data Science Agent. Analyze this dataset metadata and user request:
        User Request: {user_request}
        Dataset Summary:
        {data_schema}
        
        Generate a concrete data analysis plan and hypotheses.
        """
        plan_response = self.client.models.generate_content(
            model=self.model_name,
            contents=planning_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ANALYSIS_PLAN_SCHEMA,
                temperature=0.2
            )
        )
        plan = json.loads(plan_response.text)

        # --- STEP 2: CODE EXECUTION & PLOTTING ---
        execution_prompt = f"""
        You are a Senior Data Analyst executing code workflows in a python sandbox.
        Below is the raw CSV data:
        --- DATASTART ---
        {csv_content}
        --- DATAEND ---
        
        User's explicit target: {user_request}
        Your planned objectives: {json.dumps(plan)}
        
        CRITICAL VISUALIZATION INSTRUCTIONS:
        1. In addition to printing numerical metrics to stdout, you MUST generate exactly TWO plots using matplotlib or seaborn:
           - A Boxplot representing distributions (e.g., numeric variables across categorical categories). Save it explicitly as 'boxplot.png'.
           - A Scatter diagram representing relationships/correlations. Save it explicitly as 'scatter.png'.
        2. Make sure to use `plt.savefig('boxplot.png')` and `plt.savefig('scatter.png')`. Do not use `plt.show()`.
        3. Parse the data using `io.StringIO`.
        """
        code_response = self.client.models.generate_content(
            model=self.model_name,
            contents=execution_prompt,
            config=types.GenerateContentConfig(
                tools=[{"code_execution": {}}],
                temperature=0.1
            )
        )

        sandbox_logs = ""
        images_found = []

        for part in code_response.candidates[0].content.parts:
            if part.code_execution_result:
                sandbox_logs += part.code_execution_result.output + "\n"
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                images_found.append(part.inline_data)

        generated_charts = {}
        if len(images_found) >= 1:
            b64_1 = base64.b64encode(images_found[0].data).decode("utf-8")
            generated_charts["boxplot"] = f"data:{images_found[0].mime_type};base64,{b64_1}"
        if len(images_found) >= 2:
            b64_2 = base64.b64encode(images_found[1].data).decode("utf-8")
            generated_charts["scatter"] = f"data:{images_found[1].mime_type};base64,{b64_2}"

        # --- STEP 3: SYNTHESIZE ---
        synthesis_prompt = f"""
        Based on the data crunching phase console logs:
        {sandbox_logs if sandbox_logs else code_response.text}
        
        Synthesize outcomes into clean, structured insights matching the final requested schema.
        """
        final_response = self.client.models.generate_content(
            model=self.model_name,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FINAL_INSIGHTS_SCHEMA,
                temperature=0.2
            )
        )
        
        insights = json.loads(final_response.text)
        insights["charts"] = generated_charts
        return insights

app = FastAPI(title="Gemini Data Analytics Agent")
templates = Jinja2Templates(directory="templates")
agent = GeminiDataAgent()

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"results": None})

@app.post("/analyze", response_class=HTMLResponse)
async def handle_analysis(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        contents = await file.read()
        
        # Load dataset into pandas
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # 1. Capture pandas .info() layout using an internal memory string buffer
        buffer = io.StringIO()
        df.info(buf=buffer)
        pandas_info_str = buffer.getvalue()
        
        # 2. Capture pandas .describe() statistical summary matrix as an HTML fragment
        pandas_describe_html = df.describe(include='all').to_html(
            classes="w-full text-sm text-left text-gray-600 border-collapse border border-gray-200",
            border=0
        )
        
        # Run agent analytical routine
        insights = agent.analyze_bytes_data(df, prompt)
        
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": insights, 
                "prompt": prompt, 
                "filename": file.filename,
                "pandas_info": pandas_info_str,
                "pandas_describe": pandas_describe_html,
                "error_message": None
            }
        )
    except Exception as e:
        print(f"❌ Underlying Agent Error Trace: {str(e)}")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": None, 
                "prompt": prompt, 
                "filename": file.filename,
                "error_message": f"Agent Execution Failed: {str(e)}"
            }
        )

# import os
# import json
# import io
# import sys
# import base64
# import re
# import pandas as pd
# import matplotlib
# # Prevent matplotlib from trying to open GUI windows during server executions
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# class OpenRouterDataAgent:
#     def __init__(self, model_name: str = "openai/gpt-oss-120b:free"):
#         api_key = os.getenv("OPENROUTER_API_KEY")
#         if not api_key:
#             raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        
#         # Configure the standard OpenAI Client to point to OpenRouter
#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=api_key,
#         )
#         self.model_name = model_name

#     def analyze_dataframe(self, df: pd.DataFrame, user_request: str):
#         # Create a lightweight string layout profile of the data for context
#         data_schema = f"Columns: {list(df.columns)}\nData Types:\n{df.dtypes.to_string()}\nShape: {df.shape}"
        
#         # --- STEP 1: PLANNING & CODE GENERATION ---
#         prompt = f"""
#         You are an expert Data Scientist. Analyze this dataset metadata:
#         {data_schema}
        
#         The user wants to find out: "{user_request}"
        
#         Write a Python script to analyze this data. The data is pre-loaded into a pandas DataFrame named `df`.
        
#         CRITICAL OUTPUT REQUIREMENTS:
#         1. Your python code must print a short summary analysis statement to stdout.
#         2. Your code MUST generate exactly TWO plots using matplotlib or seaborn:
#            - A Boxplot representing distributions. Save it to 'boxplot.png'.
#            - A Scatter diagram representing correlations. Save it to 'scatter.png'.
#         3. Do not use plt.show(). Use plt.savefig('boxplot.png') and plt.savefig('scatter.png').
        
#         Respond ONLY with the executable python code inside a markdown code block. Do not write text outside the code block.
#         """
        
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1
#         )
        
#         llm_output = response.choices[0].message.content
        
#         # Extract the raw python code using regex out of the markdown blocks
#         code_match = re.search(r"```python\s*(.*?)\s*```", llm_output, re.DOTALL)
#         python_code = code_match.group(1) if code_match else llm_output

#         # Clean up any residual plot files from past execution sessions
#         for f in ['boxplot.png', 'scatter.png']:
#             if os.path.exists(f): os.remove(f)

#         # --- STEP 2: SECURE LOCAL EXECUTION ---
#         # Intercept output metrics from running the python script block locally
#         old_stdout = sys.stdout
#         redirected_output = io.StringIO()
#         sys.stdout = redirected_output
        
#         try:
#             # Execute code directly using the preloaded data structure
#             local_scope = {"df": df, "plt": plt, "sns": sns, "pd": pd}
#             exec(python_code, local_scope)
#         except Exception as e:
#             print(f"Execution Error: {str(e)}")
#         finally:
#             sys.stdout = old_stdout
            
#         sandbox_logs = redirected_output.getvalue()

#         # Read generated plots and convert to inline Base64 data strings for template rendering
#         generated_charts = {}
#         if os.path.exists('boxplot.png'):
#             with open('boxplot.png', 'rb') as img_f:
#                 generated_charts['boxplot'] = f"data:image/png;base64,{base64.b64encode(img_f.read()).decode('utf-8')}"
#         if os.path.exists('scatter.png'):
#             with open('scatter.png', 'rb') as img_f:
#                 generated_charts['scatter'] = f"data:image/png;base64,{base64.b64encode(img_f.read()).decode('utf-8')}"

#         # --- STEP 3: SYNTHESIZE VIA OPENROUTER ---
#         synthesis_prompt = f"""
#         Review this stdout log generated by our data crunching script:
#         {sandbox_logs}
        
#         Return a clean JSON object summarizing everything for the UI.
#         The JSON payload format must be:
#         {{
#             "summary_of_findings": "Detailed multi-paragraph analytical summary here.",
#             "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
#         }}
#         """
        
#         synth_response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": synthesis_prompt}],
#             response_format={"type": "json_object"} if "free" not in self.model_name else None,
#             temperature=0.2
#         )
        
#         # Parse output data carefully
#         try:
#             raw_text = synth_response.choices[0].message.content
#             # Strip markdown block wraps if model wraps JSON output block text
#             if "```" in raw_text:
#                 raw_text = re.search(r"({.*})", raw_text, re.DOTALL).group(1)
#             insights = json.loads(raw_text)
#         except Exception:
#             insights = {
#                 "summary_of_findings": synth_response.choices[0].message.content,
#                 "recommendations": ["Review dataset properties manually for null targets."]
#             }
            
#         insights["charts"] = generated_charts
#         return insights

# app = FastAPI(title="OpenRouter Data Analytics Agent")
# templates = Jinja2Templates(directory="templates")
# agent = OpenRouterDataAgent()

# @app.get("/", response_class=HTMLResponse)
# async def read_index(request: Request):
#     return templates.TemplateResponse(request=request, name="index.html", context={"results": None})

# @app.post("/analyze", response_class=HTMLResponse)
# async def handle_analysis(
#     request: Request,
#     file: UploadFile = File(...),
#     prompt: str = Form(...)
# ):
#     if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
#         raise HTTPException(status_code=400, detail="Invalid file type.")
    
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
        
#         buffer = io.StringIO()
#         df.info(buf=buffer)
#         pandas_info_str = buffer.getvalue()
        
#         pandas_describe_html = df.describe(include='all').to_html(
#             classes="w-full text-sm text-left text-gray-600 border-collapse border border-gray-200", border=0
#         )
        
#         insights = agent.analyze_dataframe(df, prompt)
        
#         return templates.TemplateResponse(
#             request=request,
#             name="index.html",
#             context={
#                 "results": insights, 
#                 "prompt": prompt, 
#                 "filename": file.filename,
#                 "pandas_info": pandas_info_str,
#                 "pandas_describe": pandas_describe_html,
#                 "error_message": None
#             }
#         )
#     except Exception as e:
#         print(f"❌ Agent Error Trace: {str(e)}")
#         return templates.TemplateResponse(
#             request=request,
#             name="index.html",
#             context={
#                 "results": None, 
#                 "prompt": prompt, 
#                 "filename": file.filename,
#                 "error_message": f"Execution Failed: {str(e)}"
#             }
#         )