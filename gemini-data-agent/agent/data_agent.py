import json
import io
import pandas as pd
from google.genai import types
from config import client, settings

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
        "key_metrics": {"type": "OBJECT", "description": "Key numbers or aggregates calculated."},
        "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["summary_of_findings", "key_metrics", "recommendations"]
}

class GeminiDataAgent:
    def __init__(self):
        self.client = client
        self.model_name = settings.model_name

    def run_analysis(self, csv_content: str, user_request: str) -> dict:
        df_snippet = pd.read_csv(io.StringIO(csv_content), nrows=5)
        data_schema = f"Columns: {list(df_snippet.columns)}\nData Types:\n{df_snippet.dtypes.to_string()}"

        # 1. Plan Phase
        planning_prompt = f"User Request: {user_request}\nDataset Summary:\n{data_schema}\nGenerate a data analysis plan."
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

        # 2. Execute Phase
        execution_prompt = f"""
        You are a Data Analyst running inside a Python sandbox. Analyze this CSV string:
        
        csv_data = '''{csv_content}'''
        
        User Target: {user_request}
        Objectives: {json.dumps(plan)}
        
        Write Python code using pandas and io.StringIO to parse csv_data, compute metrics, and print the output clearly using print().
        """
        code_response = self.client.models.generate_content(
            model=self.model_name,
            contents=execution_prompt,
            config=types.GenerateContentConfig(tools=[{"code_execution": {}}], temperature=0.1)
        )

        sandbox_logs = ""
        executed_code = ""
        
        # Guard clause check to make sure candidates returned safely
        if code_response.candidates and code_response.candidates[0].content.parts:
            for part in code_response.candidates[0].content.parts:
                if part.executable_code:
                    executed_code += part.executable_code.code + "\n"
                if part.code_execution_result:
                    sandbox_logs += part.code_execution_result.output + "\n"

        # 3. Synthesize Phase
        synthesis_input = sandbox_logs if sandbox_logs.strip() else code_response.text
        synthesis_prompt = f"Based on these sandbox execution outputs:\n{synthesis_input}\nSynthesize final insights matching the requested schema."
        
        final_response = self.client.models.generate_content(
            model=self.model_name,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FINAL_INSIGHTS_SCHEMA,
                temperature=0.2
            )
        )
        
        return {
            "plan": plan,
            "code": executed_code,
            "logs": sandbox_logs,
            "insights": json.loads(final_response.text)
        }