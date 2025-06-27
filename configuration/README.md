# Smart Quiz Generator Microservice

## 1. Objective

This microservice generates goal-aligned quizzes. It accepts a user goal (e.g., "GATE ECE", "Amazon SDE"), difficulty, and other optional parameters, returning a JSON array of multiple-choice and/or short-answer questions. Questions are dynamically generated from local data sources and offline models (TF-IDF retrieval and T5-based generation).

## 2. Prerequisites

*   Docker Desktop installed and running.
*   A terminal or command prompt (e.g., Command Prompt, PowerShell, Git Bash, Linux/macOS terminal).
*   The project files structured with a parent directory containing `smart-quiz/` (with application code) and `configuration/` (with Dockerfile, this README, etc.) as sibling directories.

## 3. Setup & Running the Service (Using Docker)

These commands should be run from the **parent directory** that contains both the `smart-quiz/` and `configuration/` folders (referred to as `PROJECT_PARENT_DIR` below).

### 3.1. Build the Docker Image

Navigate to `PROJECT_PARENT_DIR` in your terminal and run:

```bash
docker build --no-cache -t smart-quiz-service -f configuration/Dockerfile .


--no-cache: Ensures a fresh build, especially useful if requirements.txt has changed.

-t smart-quiz-service: Tags the image as smart-quiz-service.

-f configuration/Dockerfile: Specifies the path to the Dockerfile (which is inside the configuration subdirectory of your build context).

.: Uses the current directory (PROJECT_PARENT_DIR) as the Docker build context.

3.2. Run the Docker Container

Once the image is built, run the container from PROJECT_PARENT_DIR:

For Windows Command Prompt (cmd.exe):

docker run -d -p 8000:8000 ^
  -v "%cd%\smart-quiz\config.json:/app/config.json" ^
  -v "%cd%\smart-quiz\data:/app/data" ^
  -v "%cd%\smart-quiz\app\model:/app/app/model" ^
  --name smart-quiz-container ^
  smart-quiz-service
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Cmd
IGNORE_WHEN_COPYING_END

For PowerShell:

docker run -d -p 8000:8000 `
  -v "$(Get-Location)\smart-quiz\config.json:/app/config.json" `
  -v "$(Get-Location)\smart-quiz\data:/app/data" `
  -v "$(Get-Location)\smart-quiz\app\model:/app/app/model" `
  --name smart-quiz-container `
  smart-quiz-service
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Powershell
IGNORE_WHEN_COPYING_END

For Linux/macOS/Git Bash:

docker run -d -p 8000:8000 \
  -v "$(pwd)/smart-quiz/config.json:/app/config.json" \
  -v "$(pwd)/smart-quiz/data:/app/data" \
  -v "$(pwd)/smart-quiz/app/model:/app/app/model" \
  --name smart-quiz-container \
  smart-quiz-service
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Explanation of docker run options:

-d: Runs the container in detached mode (in the background).

-p 8000:8000: Maps port 8000 on your host machine to port 8000 inside the container.

-v <host_path>:<container_path>: Mounts a local directory or file into the container. This is useful for development, allowing you to change config.json, data, or models locally and have those changes reflected inside the running service without rebuilding the image.

--name smart-quiz-container: Assigns a memorable name to the running container.

smart-quiz-service: The name of the Docker image to run.

3.3. Accessing the Service

Once the container is running, the API is available at http://localhost:8000.

API Documentation (Swagger UI): http://localhost:8000/docs

Health Check: http://localhost:8000/health

Version Info: http://localhost:8000/version

3.4. Viewing Logs

To see the application logs from the Docker container:

docker logs smart-quiz-container
# To follow logs in real-time:
docker logs -f smart-quiz-container
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
3.5. Stopping and Removing the Container
docker stop smart-quiz-container
docker rm smart-quiz-container
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
4. API Endpoints

Refer to the Swagger UI at http://localhost:8000/docs for interactive documentation and to try out the API.
The formal API schemas are also defined in configuration/schema.json.

4.1. Generate Quiz (POST /generate)

Description: Generates a quiz based on the provided criteria.

Content-Type: application/json

Request Body Parameters:

goal (string, required): The learning goal (e.g., "Amazon SDE", "GATE ECE").

num_questions (integer, optional): Number of questions desired. Defaults to value in config.json. Must be > 0.

difficulty (string, required): Difficulty level. Must be one of the values in supported_difficulties from config.json (e.g., "beginner", "intermediate", "advanced").

topic (string, optional): Specific topic within the goal (e.g., "Data Structures", "SQL").

type (string, optional): Desired question type. Must be one of the values in supported_types from config.json (e.g., "mcq", "short_answer").

generator_mode (string, optional): Overrides the server's default generation strategy. Must be one of the values in allowed_generator_modes from config.json (e.g., "hybrid", "retrieval", "t5_only", "template").

Example curl Request:

curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "goal": "Amazon SDE",
  "num_questions": 3,
  "difficulty": "intermediate",
  "type": "mcq",
  "generator_mode": "hybrid"
}'
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Example Success Response (200 OK):

{
  "quiz_id": "quiz_abcdef12",
  "goal": "Amazon SDE",
  "questions": [
    {
      "type": "mcq",
      "question": "What is the primary purpose of a hash table?",
      "options": {
        "A": "To store elements in a sorted order.",
        "B": "To provide fast key-based lookups, insertions, and deletions.",
        "C": "To ensure data immutability.",
        "D": "To manage hierarchical data."
      },
      "answer": "B",
      "difficulty": "intermediate",
      "topic": "Data Structures",
      "goal": "Amazon SDE",
      "generation_method": "retrieval_tfidf",
      "t5_template_id": null
    }
    // ... more questions
  ]
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

Common Error Responses:

400 Bad Request: If difficulty or type is unsupported, or generator_mode is invalid.

404 Not Found: If no questions exist in the bank for the specified goal.

422 Unprocessable Entity: If the request JSON is malformed or misses required fields like goal or difficulty.

500 Internal Server Error: If the server encounters an issue generating questions for valid criteria. Check server logs for details.

4.2. Health Check (GET /health)

Description: Returns the health status of the service.

Example Response: {"status": "ok", "message": "Service is healthy."}

4.3. Version Information (GET /version)

Description: Returns version and configuration details of the running service.

Example Response: (Content will depend on your config.json)

{
  "version": "1.0.1-logging",
  "default_generator_mode_from_config": "hybrid",
  "model_info": "TF-IDF (sklearn), T5-small (Hugging Face Transformers)",
  "question_bank_size": 200,
  "supported_difficulties": ["beginner", "intermediate", "advanced"],
  "supported_types": ["mcq", "short_answer"],
  "allowed_generator_modes_for_request": ["hybrid", "retrieval", "retrieval_only", "t5_only", "template"]
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
5. Configuration (smart-quiz/config.json)

The primary configuration for the service is managed through the smart-quiz/config.json file. If running the Docker container with the volume mount for config.json (as shown in section 3.2), you can edit this file locally. A restart of the Docker container is required for changes to config.json to take effect.

Key configurable parameters:

generator_mode (string): The default strategy for question generation.

"hybrid": Attempts retrieval first, then uses T5 for remaining questions.

"retrieval" or "retrieval_only": Uses only the TF-IDF based retrieval from the question bank.

"t5_only" or "template": Uses only T5-based generation (which relies on templates defined in smart-quiz/data/templates.json).

version (string): Application version displayed by the /version endpoint.

default_num_questions (integer): Default number of questions if not specified in the request.

max_questions (integer): Maximum number of questions that can be requested.

supported_difficulties (array of strings): List of valid difficulty levels.

supported_types (array of strings): List of valid question types.

allowed_generator_modes (array of strings): List of modes that can be specified in the API request's generator_mode field.

logging_level (string): Controls the verbosity of application logs (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").

t5_generation_params (object): Parameters passed to the T5 model pipeline for text generation (e.g., max_length, num_beams, temperature).

6. Project Structure Overview

PROJECT_PARENT_DIR/

smart-quiz/: Contains the core application Python code, data, models, and runtime configuration.

app/: FastAPI application (main.py), Pydantic models, and generation logic (generator.py).

app/model/: Stores the pre-trained TF-IDF vectorizer (.pkl file).

data/: Stores the question_bank.json and templates.json.

config.json: Main runtime configuration for the service.

configuration/: Contains Docker-related files, schema definitions, this README, and tests.

Dockerfile: Instructions to build the Docker image.

requirements.txt: Python dependencies.

schema.json: JSON Schema definitions for API inputs/outputs.

README.md: This file.

tests/: Automated tests (e.g., test_generate.py).

7. Development Notes

The application uses offline models: TF-IDF for retrieval and a local T5-small model for generative tasks.

The quality of T5-generated questions heavily depends on the quality of seed questions in question_bank.json and the design of prompts in templates.json.

The QuizGeneratorT5._extract_data_for_template method in smart-quiz/app/generator.py requires specific Python logic for each template_id in templates.json to correctly extract data from seed questions. Ensure this is implemented for all desired templates.

For debugging, set logging_level: "DEBUG" in config.json and restart the container to see detailed logs.

---
Remember to replace generic paths like `PROJECT_PARENT_DIR` or `C:\Users\YourUser\Path\To\ProjectParent` with the actual path if you are sharing this document in a context where that's fixed. Otherwise, the instruction to run commands from the "project parent directory" is clear.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END