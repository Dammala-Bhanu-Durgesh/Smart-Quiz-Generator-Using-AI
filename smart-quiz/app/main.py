import os
import json
import uuid
import sys
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from .generator import QuizMaster, ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE, set_config_for_generator

APP_CONFIG: Dict[str, Any] = {}
MAIN_PY_DIR = os.path.dirname(os.path.abspath(__file__))
SMART_QUIZ_DIR_FOR_CONFIG = os.path.dirname(MAIN_PY_DIR)
CONFIG_PATH = os.path.join(SMART_QUIZ_DIR_FOR_CONFIG, "config.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
main_logger = logging.getLogger("smart_quiz.main")

def load_app_config_for_main():
    global APP_CONFIG
    if not os.path.exists(CONFIG_PATH):
        print(f"CRITICAL: config.json not found at {CONFIG_PATH}. Terminating.")
        sys.exit(1)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: APP_CONFIG = json.load(f)
        log_level_str = APP_CONFIG.get("logging_level", "INFO").upper()
        numeric_log_level = getattr(logging, log_level_str, logging.INFO)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(
            level=numeric_log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)], force=True
        )
        main_logger.info(f"Config loaded from {CONFIG_PATH}. Log level: {log_level_str}.")
        APP_CONFIG.setdefault("generator_mode", "hybrid")
        APP_CONFIG.setdefault("version", "0.0.0-default")
        APP_CONFIG.setdefault("default_num_questions", 5)
        APP_CONFIG.setdefault("max_questions", 10)
        APP_CONFIG.setdefault("supported_difficulties", ["beginner", "intermediate", "advanced"])
        APP_CONFIG.setdefault("supported_types", ["mcq", "short_answer"])
        APP_CONFIG.setdefault("model_info", "Default Model Info")
        APP_CONFIG.setdefault("t5_generation_params", {})
        critical_keys = ["generator_mode", "max_questions", "supported_difficulties", "supported_types", "version"]
        for key in critical_keys:
            if key not in APP_CONFIG:
                main_logger.critical(f"CRITICAL CONFIG ERROR: Key '{key}' missing. Terminating.")
                sys.exit(1)
    except Exception as e:
        print(f"CRITICAL: Error loading config: {e}. Terminating.")
        sys.exit(1)
    set_config_for_generator(APP_CONFIG)

load_app_config_for_main()

app = FastAPI(
    title="Smart Quiz Generator",
    version=APP_CONFIG.get("version"),
    description="Generates quizzes based on user goals, difficulty, and other criteria."
)

class QuizGenerationRequest(BaseModel):
    goal: str = Field(..., example="Amazon SDE")
    num_questions: int = Field(default_factory=lambda: APP_CONFIG.get("default_num_questions", 5), gt=0, example=5)
    difficulty: str = Field(..., example="intermediate")
    topic: Optional[str] = Field(None, example=None, description="Optional: The specific topic.")
    type: Optional[str] = Field(None, example=None, description="Optional: 'mcq' or 'short_answer'.")

class QuestionResponse(BaseModel): # OUTPUT SCHEMA CHANGE
    type: str
    question: str
    options: Optional[List[str]] = None
    answer: str
    difficulty: str
    topic: Optional[str] = None
    # goal: Optional[str] = None  # <-- REMOVED from individual question response

class QuizResponse(BaseModel):
    quiz_id: str
    goal: str # Overall goal for the quiz is still here
    questions: List[QuestionResponse]

quiz_master_instance: Optional[QuizMaster] = None

@app.on_event("startup")
async def startup_event():
    global quiz_master_instance
    main_logger.info("FastAPI application startup event.")
    quiz_master_instance = QuizMaster(app_config_from_main=APP_CONFIG)
    main_logger.info("QuizMaster initialized successfully.")

@app.post("/generate", response_model=QuizResponse, response_model_exclude_none=True, tags=["Quiz Generation"])
async def generate_quiz_api(request: QuizGenerationRequest):
    main_logger.info(f"Received /generate request: {request.dict(exclude_none=True)}")
    if quiz_master_instance is None: main_logger.error("QuizMaster not available."); raise HTTPException(status_code=503, detail="Service not ready.")

    if request.difficulty not in APP_CONFIG.get("supported_difficulties", []):
        main_logger.warning(f"Unsupported difficulty: {request.difficulty}"); raise HTTPException(status_code=400, detail="Unsupported difficulty.")
    if request.type and request.type not in APP_CONFIG.get("supported_types", []):
        main_logger.warning(f"Unsupported type: {request.type}"); raise HTTPException(status_code=400, detail="Unsupported type.")

    num_questions_req = request.num_questions
    max_q = APP_CONFIG.get("max_questions", 10)
    if num_questions_req > max_q: num_questions_req = max_q; main_logger.info(f"Num_questions capped at {max_q}")

    effective_generator_mode = APP_CONFIG.get("generator_mode", "hybrid").lower()
    main_logger.info(f"Using server-configured generator_mode: '{effective_generator_mode}'")

    try:
        generated_question_list_dicts = quiz_master_instance.generate_quiz_hybrid(
            goal=request.goal,
            num_questions_total=num_questions_req,
            difficulty=request.difficulty,
            topic=request.topic,
            q_type=request.type
        )
    except Exception as e_gen:
        main_logger.error(f"Error in quiz_master.generate_quiz_hybrid: {e_gen}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during question generation.")

    if not generated_question_list_dicts and num_questions_req > 0:
        main_logger.warning(f"Failed to generate questions for {request.goal}, mode {effective_generator_mode}")
        source_exists = any(q.get('goal_from_file','').lower() == request.goal.lower() for q in ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE)
        if not source_exists: raise HTTPException(status_code=404, detail=f"No questions for goal '{request.goal}'.")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions for goal '{request.goal}'.")

    pydantic_questions = []
    for q_dict_from_generator in generated_question_list_dicts:
        # Explicitly construct the data for QuestionResponse to ensure only allowed fields are passed
        response_q_data = {
            "type": q_dict_from_generator.get("type"),
            "question": q_dict_from_generator.get("question"),
            "options": q_dict_from_generator.get("options"),
            "answer": q_dict_from_generator.get("answer"),
            "difficulty": q_dict_from_generator.get("difficulty"),
            "topic": q_dict_from_generator.get("topic")
            # 'goal' is no longer included here as it's not in QuestionResponse model
        }
        try:
            pydantic_questions.append(QuestionResponse(**response_q_data))
        except Exception as e_pydantic_val:
            main_logger.error(f"Pydantic validation error for question data: {response_q_data}, Error: {e_pydantic_val}", exc_info=True)
            continue

    main_logger.info(f"Successfully prepared {len(pydantic_questions)} questions for response.")
    return QuizResponse(
        quiz_id=f"quiz_{uuid.uuid4().hex[:8]}",
        goal=request.goal, # Overall quiz goal
        questions=pydantic_questions
    )

@app.get("/health", tags=["Utilities"])
async def health_check():
    main_logger.debug("Health check accessed.")
    if quiz_master_instance and quiz_master_instance.retrieval_generator: return {"status": "ok", "message": "Service healthy."}
    main_logger.warning("Health check: DEGRADED/ERROR"); return {"status": "degraded", "message": "Component issue."}

@app.get("/version", tags=["Utilities"])
async def version_info():
    main_logger.debug("Version endpoint accessed.")
    return {"version": APP_CONFIG.get("version"), "configured_generator_mode": APP_CONFIG.get("generator_mode"), "model_info": APP_CONFIG.get("model_info"), "question_bank_size": len(ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE) if ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE is not None else 0, "supported_difficulties": APP_CONFIG.get("supported_difficulties"), "supported_types": APP_CONFIG.get("supported_types")}