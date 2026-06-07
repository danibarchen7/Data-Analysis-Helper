from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.data_agent import GeminiDataAgent

router = APIRouter(prefix="/api/v1")
agent = GeminiDataAgent()

@router.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are currently supported.")
    
    try:
        contents = await file.read()
        csv_content = contents.decode("utf-8")
        return agent.run_analysis(csv_content, prompt)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))