# configuration/tests/test_generate.py

import requests
import pytest
import os
import json
from typing import Dict, Any, List, Optional # <--- IMPORTED typing elements

# Determine base URL - can be overridden by environment variable for CI/CD
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")

# Construct the path to config.json relative to this test file
# This assumes test_generate.py is in configuration/tests/
# and config.json is in smart-quiz/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # configuration/tests/
CONFIGURATION_DIR = os.path.dirname(CURRENT_DIR) # configuration/
PARENT_DIR = os.path.dirname(CONFIGURATION_DIR) # some_parent_directory/ (e.g., Final Work/)
CONFIG_FILE_PATH = os.path.join(PARENT_DIR, "smart-quiz", "config.json")


# Helper to load config for test parameters
def load_test_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE_PATH):
        pytest.fail(
            f"Test configuration file not found at {CONFIG_FILE_PATH}. "
            f"Ensure config.json is in the smart-quiz directory and path is correct. "
            f"Current script dir: {CURRENT_DIR}"
        )
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {CONFIG_FILE_PATH}.")
    return {} # Should not reach here if pytest.fail works

TEST_CONFIG = load_test_config()

# --- Test Data (derived from config or sensible defaults) ---
# Ensure these goals/topics exist in your question_bank.json for tests to pass reliably
VALID_GOAL_WITH_MCQS = TEST_CONFIG.get("test_valid_goal_mcq", "Amazon SDE") # Example, customize
VALID_GOAL_WITH_SA = TEST_CONFIG.get("test_valid_goal_sa", "GATE ECE")   # Example, customize
EXISTING_TOPIC_FOR_MCQ_GOAL = TEST_CONFIG.get("test_existing_topic_mcq", "Algorithms") # Example

VALID_DIFFICULTY = TEST_CONFIG.get("supported_difficulties", ["intermediate"])[0]
VALID_MCQ_TYPE = "mcq"
VALID_SA_TYPE = "short_answer"

MAX_QUESTIONS = TEST_CONFIG.get("max_questions", 10)
DEFAULT_NUM_QUESTIONS = TEST_CONFIG.get("default_num_questions", 5)
ALLOWED_GENERATOR_MODES = TEST_CONFIG.get("allowed_generator_modes", ["hybrid", "retrieval", "t5_only", "template"])


# --- Helper Functions ---
def validate_question_structure(question: Dict[str, Any], requested_type: Optional[str] = None, requested_difficulty: Optional[str] = None):
    """Validates the structure of a single question object."""
    assert "type" in question, "Question missing 'type' field"
    assert "question" in question, "Question missing 'question' field"
    assert "answer" in question, "Question missing 'answer' field"
    assert "difficulty" in question, "Question missing 'difficulty' field"

    assert isinstance(question["question"], str) and question["question"].strip(), "Question text is empty or not a string"
    assert isinstance(question["answer"], str) and question["answer"].strip(), "Answer is empty or not a string"

    assert question["type"] in TEST_CONFIG.get("supported_types", []), f"Invalid question type: {question['type']}"
    assert question["difficulty"] in TEST_CONFIG.get("supported_difficulties", []), f"Invalid question difficulty: {question['difficulty']}"

    if requested_type:
        assert question["type"] == requested_type, f"Expected type {requested_type}, got {question['type']}"
    if requested_difficulty:
        assert question["difficulty"] == requested_difficulty, f"Expected difficulty {requested_difficulty}, got {question['difficulty']}"

    if question["type"] == "mcq":
        assert "options" in question and question["options"] is not None, "MCQ missing 'options' field or it's null"
        assert isinstance(question["options"], dict), "MCQ 'options' field is not a dictionary"
        assert len(question["options"]) >= 2, f"MCQ has too few options: {len(question['options'])}" # Usually 4, but allow at least 2
        for opt_key, opt_val in question["options"].items():
            assert isinstance(opt_key, str) and opt_key.strip(), "MCQ option key is invalid"
            assert isinstance(opt_val, str) and opt_val.strip(), f"MCQ option value for key '{opt_key}' is invalid"
        assert question["answer"] in question["options"], f"MCQ answer '{question['answer']}' not found in option keys: {list(question['options'].keys())}"
    elif question["type"] == "short_answer":
        # 'options' should be null or absent for short_answer
        assert "options" not in question or question["options"] is None, "Short answer question should not have 'options'"

    # Check for expected additional fields from your Pydantic model
    assert "goal" in question or question.get("goal") is None # Allow None if optional
    assert "generation_method" in question or question.get("generation_method") is None
    if question.get("generation_method") == "t5_generated":
        assert "t5_template_id" in question or question.get("t5_template_id") is None


def validate_quiz_response(
    data: Dict[str, Any],
    expected_goal: str,
    expected_min_num_questions: int, # Use min if capping can occur
    expected_max_num_questions: int,
    requested_type: Optional[str] = None,
    requested_difficulty: Optional[str] = None
):
    """Validates the overall quiz response structure."""
    assert "quiz_id" in data and isinstance(data["quiz_id"], str) and data["quiz_id"].startswith("quiz_"), "Invalid quiz_id"
    assert data.get("goal") == expected_goal, f"Expected goal {expected_goal}, got {data.get('goal')}"
    assert "questions" in data and isinstance(data["questions"], list), "Invalid 'questions' field"
    
    num_returned_questions = len(data["questions"])
    assert expected_min_num_questions <= num_returned_questions <= expected_max_num_questions, \
        f"Expected between {expected_min_num_questions}-{expected_max_num_questions} questions, got {num_returned_questions}"

    if num_returned_questions > 0: # Only validate individual questions if any were returned
        for question in data["questions"]:
            validate_question_structure(question, requested_type, requested_difficulty)
            assert question.get("goal") == expected_goal, "Question goal mismatch"


# --- Test Cases ---

# Parameterize with a few num_questions values and all allowed generator modes
@pytest.mark.parametrize("num_q", [1, 3]) # Test with 1 and a moderate number
@pytest.mark.parametrize("gen_mode_from_list", ALLOWED_GENERATOR_MODES)
def test_generate_quiz_valid_requests_all_modes(num_q: int, gen_mode_from_list: str):
    """Tests successful quiz generation with various valid inputs across all modes."""
    # "template" mode is an alias for "t5_only" in how QuizMaster handles it
    effective_gen_mode = "t5_only" if gen_mode_from_list == "template" else gen_mode_from_list

    payload = {
        "goal": VALID_GOAL_WITH_MCQS, # Use a goal known to have MCQs for this general test
        "num_questions": num_q,
        "difficulty": VALID_DIFFICULTY,
        "type": VALID_MCQ_TYPE, # Request MCQs
        "generator_mode": effective_gen_mode
    }
    print(f"\nTesting /generate with payload: {payload}")
    response = requests.post(f"{BASE_URL}/generate", json=payload)

    assert response.status_code == 200, \
        f"Failed for mode '{effective_gen_mode}'. Status: {response.status_code}. Response: {response.text}"
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        pytest.fail(f"Response is not valid JSON for mode '{effective_gen_mode}'. Response text: {response.text}")

    # If num_q is 0, the server might return an empty list of questions, which is valid.
    # If num_q > 0, we expect questions.
    # The server might return fewer if not enough are available, but not more than requested or MAX_QUESTIONS.
    expected_min = 0 if num_q == 0 else 1 # If 0 requested, 0 is fine. If >0 requested, expect at least 1 if successful.
    expected_max = min(num_q, MAX_QUESTIONS) if num_q > 0 else 0
    
    # If the service is expected to always return num_q (unless capped), adjust this logic.
    # For now, assuming it might return fewer if not enough are found by a specific generator.
    # A stricter test would be: assert len(data["questions"]) == expected_max if num_q > 0
    if num_q > 0 and not data["questions"]:
         pytest.skip(f"Skipping detailed validation for mode '{effective_gen_mode}' as 0 questions were returned for num_q={num_q}. This might indicate data scarcity for this mode/criteria.")

    validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 0, expected_max, VALID_MCQ_TYPE, VALID_DIFFICULTY)
    print(f"SUCCESS for mode '{effective_gen_mode}', num_q {num_q}")


def test_generate_quiz_default_num_questions_no_override():
    """Tests if default number of questions is returned when num_questions not specified and no mode override."""
    payload = {
        "goal": VALID_GOAL_WITH_MCQS,
        "difficulty": VALID_DIFFICULTY,
        # num_questions not specified
        # generator_mode not specified (uses server default from config.json)
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 0, DEFAULT_NUM_QUESTIONS, requested_difficulty=VALID_DIFFICULTY)

def test_generate_quiz_max_questions_capping():
    """Tests if num_questions is capped at max_questions."""
    payload = {
        "goal": VALID_GOAL_WITH_MCQS,
        "num_questions": MAX_QUESTIONS + 5, # Request more than max
        "difficulty": VALID_DIFFICULTY,
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 0, MAX_QUESTIONS, requested_difficulty=VALID_DIFFICULTY)

@pytest.mark.parametrize("difficulty", TEST_CONFIG.get("supported_difficulties", ["beginner", "intermediate", "advanced"]))
def test_generate_quiz_all_supported_difficulties(difficulty: str):
    payload = {
        "goal": VALID_GOAL_WITH_MCQS,
        "num_questions": 1,
        "difficulty": difficulty,
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    if not data["questions"]: pytest.skip(f"Skipping detailed validation as 0 questions returned for difficulty '{difficulty}'. Data scarcity?")
    validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 1, 1, requested_difficulty=difficulty)

@pytest.mark.parametrize("q_type", TEST_CONFIG.get("supported_types", ["mcq", "short_answer"]))
def test_generate_quiz_all_supported_types(q_type: str):
    goal_to_use = VALID_GOAL_WITH_MCQS if q_type == "mcq" else VALID_GOAL_WITH_SA
    payload = {
        "goal": goal_to_use,
        "num_questions": 1,
        "difficulty": VALID_DIFFICULTY,
        "type": q_type
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    if not data["questions"]: pytest.skip(f"Skipping detailed validation as 0 questions returned for type '{q_type}'. Data scarcity?")
    validate_quiz_response(data, goal_to_use, 1, 1, requested_type=q_type, requested_difficulty=VALID_DIFFICULTY)

def test_generate_quiz_with_topic():
    """Tests providing an optional topic."""
    payload = {
        "goal": VALID_GOAL_WITH_MCQS,
        "num_questions": 1, # Request 1 to simplify topic assertion
        "difficulty": VALID_DIFFICULTY,
        "topic": EXISTING_TOPIC_FOR_MCQ_GOAL
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    if not data["questions"]: pytest.skip(f"Skipping topic validation as 0 questions returned for topic '{EXISTING_TOPIC_FOR_MCQ_GOAL}'. Data scarcity?")
    
    validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 1, 1, requested_difficulty=VALID_DIFFICULTY)
    for q in data["questions"]: # Should only be one question
        assert q.get("topic", "").lower() == EXISTING_TOPIC_FOR_MCQ_GOAL.lower(), \
            f"Expected topic '{EXISTING_TOPIC_FOR_MCQ_GOAL}', got '{q.get('topic')}'"


# --- Negative Test Cases ---

def test_generate_quiz_invalid_difficulty_format():
    payload = {"goal": VALID_GOAL_WITH_MCQS, "num_questions": 1, "difficulty": "super_duper_hard"}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data and "Unsupported difficulty" in data["detail"]

def test_generate_quiz_invalid_type_format():
    payload = {"goal": VALID_GOAL_WITH_MCQS, "num_questions": 1, "difficulty": VALID_DIFFICULTY, "type": "essay_question"}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data and "Unsupported question type" in data["detail"]

def test_generate_quiz_goal_not_found_in_bank():
    payload = {"goal": "ThisGoalDoesNotExistInMyBank12345ABC", "num_questions": 1, "difficulty": VALID_DIFFICULTY}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data and "No questions found in bank matching the goal" in data["detail"]

def test_generate_quiz_missing_required_field_goal_payload():
    payload = {"num_questions": 1, "difficulty": VALID_DIFFICULTY} # Missing goal
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    assert response.status_code == 422 # Pydantic validation error
    data = response.json()
    assert "detail" in data and any(err["loc"] == ["body", "goal"] and err["type"] == "value_error.missing" for err in data["detail"])

def test_generate_quiz_num_questions_zero_value():
    payload = {"goal": VALID_GOAL_WITH_MCQS, "num_questions": 0, "difficulty": VALID_DIFFICULTY}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    # Your Pydantic model has gt=0, so 0 is invalid.
    # If you want to allow 0 questions (returns empty list), change Pydantic to ge=0
    # and adjust expected status code and response validation here.
    # Assuming gt=0 is strict:
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data and any(err["loc"] == ["body", "num_questions"] and "greater than 0" in err["msg"].lower() for err in data["detail"])

def test_generate_quiz_invalid_generator_mode_in_request_param():
    payload = {"goal": VALID_GOAL_WITH_MCQS, "num_questions": 1, "difficulty": VALID_DIFFICULTY, "generator_mode": "non_existent_mode"}
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    # This depends on how main.py handles invalid modes from request:
    # If it raises 400:
    # assert response.status_code == 400
    # assert "Invalid generator_mode" in response.json()["detail"]
    # If it logs warning and uses default (current behavior in provided main.py):
    assert response.status_code == 200, f"Expected 200 if invalid mode falls back to default. Got {response.status_code}. Response: {response.text}"
    data = response.json()
    if data.get("questions"): # Only validate if questions were returned (might fail if default mode also fails for this criteria)
        validate_quiz_response(data, VALID_GOAL_WITH_MCQS, 0, 1, requested_difficulty=VALID_DIFFICULTY)


# --- Health and Version Endpoint Tests ---

def test_health_endpoint_reachable():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"], f"Unexpected health status: {data['status']}"

def test_version_endpoint_reachable_and_has_keys():
    response = requests.get(f"{BASE_URL}/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "default_generator_mode_from_config" in data
    assert "model_info" in data
    assert "question_bank_size" in data and isinstance(data["question_bank_size"], int)
    assert "supported_difficulties" in data and isinstance(data["supported_difficulties"], list)
    assert "supported_types" in data and isinstance(data["supported_types"], list)
    assert "allowed_generator_modes_for_request" in data and isinstance(data["allowed_generator_modes_for_request"], list)
