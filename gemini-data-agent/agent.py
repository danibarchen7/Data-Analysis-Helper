import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
# 1. Define Native Type-Safe Schemas Using JSON-Schema Specifications
ANALYSIS_PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "investigation_goals": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "What the agent intends to discover from the dataset."
        },
        "hypotheses": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Initial assumptions or patterns to look for."
        },
        "planned_steps": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Step-by-step approach (e.g., handling missing data, grouping, computing)."
        }
    },
    "required": ["investigation_goals", "hypotheses", "planned_steps"]
}

FINAL_INSIGHTS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary_of_findings": {
            "type": "STRING",
            "description": "A comprehensive summary of the analytical insights discovered."
        },
        "key_metrics": {
            "type": "OBJECT",
            "description": "Key numbers, averages, or aggregates calculated."
        },
        "recommendations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Actionable data-driven next steps based on findings."
        }
    },
    "required": ["summary_of_findings", "key_metrics", "recommendations"]
}

# 2. The Core Gemini Data Agent Class
class GeminiDataAgent:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initializes the Gemini Client."""
        # Hardcode your key here for quick local testing (Don't commit this key to public GitHub!)
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def analyze_dataset(self, file_path: str, user_request: str):
        """Runs an end-to-end multi-step analysis on a dataset."""
        print(f"🚀 Loading dataset: {file_path}...")
        
        import pandas as pd
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        
        # Capture schema footprint and full CSV raw string data
        data_schema = f"""
        Columns: {list(df.columns)}
        Data Types:\n{df.dtypes.to_string()}
        Shape: {df.shape}
        """
        csv_content = df.to_csv(index=False)

        # --- STEP 1: INITIAL PLANNING ---
        print("\n🧠 Step 1: Agent is formulating an analysis plan...")
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
        print(f"📋 Goals: {plan.get('investigation_goals', [])}")
        print(f"📋 Steps: {plan.get('planned_steps', [])}")

        # --- STEP 2: CODE EXECUTION & DATA CRUNCHING ---
        print("\n⚙️ Step 2: Running code execution to crunch numbers...")
        
        execution_prompt = f"""
        You are a Senior Data Analyst executing a code-based workflow inside an isolated python sandbox.
        
        Below is the raw CSV data you must analyze:
        --- DATASTART ---
        {csv_content}
        --- DATAEND ---
        
        User's explicit target: {user_request}
        Your planned objectives: {json.dumps(plan)}
        
        CRITICAL RUNTIME INSTRUCTIONS:
        Write a Python script that parses the data from the string block above using `io.StringIO`, 
        computes the required metrics, aggregates rows, and prints the numerical outcomes directly 
        to the standard output console.
        
        Example structure to copy and adapt:
        ```python
        import pandas as pd
        import io
        
        csv_data = '''{csv_content}'''
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Compute things and look at stdout prints
        print("AVERAGES:")
        print(df.groupby('Product')['Sales_USD'].mean())
        ```
        """

        code_response = self.client.models.generate_content(
            model=self.model_name,
            contents=execution_prompt,
            config=types.GenerateContentConfig(
                tools=[{"code_execution": {}}],
                temperature=0.1
            )
        )

        # Track tool execution stdout output chunks
        sandbox_logs = ""
        for part in code_response.candidates[0].content.parts:
            if part.executable_code:
                print(f"\n💻 Executed Code:\n{part.executable_code.code}")
            if part.code_execution_result:
                print(f"\n🖥️ Sandbox Output:\n{part.code_execution_result.output}")
                sandbox_logs += part.code_execution_result.output + "\n"

        # --- STEP 3: SYNTHESIZE KEY FINDINGS ---
        print("\n📊 Step 3: Synthesizing final structured insights...")
        synthesis_prompt = f"""
        Based on the code execution results and terminal logs generated during your data crunching phase:
        {sandbox_logs if sandbox_logs else code_response.text}
        
        Synthesize the final outcomes into clean, structured insights matching the requested schema.
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
        
        return json.loads(final_response.text)

# 3. Execution Wrapper
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Generate the sample dataset fresh locally
    dummy_data = {
        "Date": pd.date_range(start="2026-01-01", periods=100, freq="D"),
        "Product": np.random.choice(["Laptop", "Smartphone", "Tablet", "Smartwatch"], 100),
        "Sales_USD": np.random.uniform(100, 1500, 100),
        "Units_Sold": np.random.randint(1, 10, 100),
        "Region": np.random.choice(["North", "South", "East", "West"], 100)
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv("sample_sales.csv", index=False)
    
    agent = GeminiDataAgent(model_name="gemini-2.5-flash")
    prompt = "Find which product performs best across regions and show me if there's any correlation between units sold and revenue generated."
    
    results = agent.analyze_dataset(file_path="sample_sales.csv", user_request=prompt)
    
    print("\n================ FINAL INSIGHTS ================")
    print(json.dumps(results, indent=2))