import os
import json
import uuid
import random
import re
import pickle
import logging
from typing import List, Optional, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

gen_logger = logging.getLogger("smart_quiz.generator")

APP_DIR_G = os.path.dirname(os.path.abspath(__file__))
SMART_QUIZ_BASE_DIR_G = os.path.dirname(APP_DIR_G)
QUESTION_BANK_PATH_G = os.path.join(SMART_QUIZ_BASE_DIR_G, "data", "question_bank.json")
TEMPLATES_PATH_G = os.path.join(SMART_QUIZ_BASE_DIR_G, "data", "templates.json")
TFIDF_VECTORIZER_PATH_G = os.path.join(APP_DIR_G, "model", "tfidf_vectorizer.pkl")
MODEL_NAME_QG_G = 't5-small'
APP_CONFIG_GENERATOR_SCOPE: Dict[str, Any] = {}

def set_config_for_generator(config: dict):
    global APP_CONFIG_GENERATOR_SCOPE; APP_CONFIG_GENERATOR_SCOPE = config
    gen_logger.info("Configuration received by generator module.")

def load_and_prepare_questions(path=QUESTION_BANK_PATH_G) -> List[Dict[str, Any]]:
    # This function MUST ensure that for MCQs, 'options_list' is a List[str]
    # and 'answer_text' is the TEXT of the correct option.
    # (Using the robust version from previous full code block)
    prepared_questions = []
    if not os.path.exists(path): gen_logger.error(f"Source question bank file not found at {path}."); return []
    try:
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    except Exception as e: gen_logger.error(f"Loading/decoding question bank {path}: {e}", exc_info=True); return []
    raw_questions = data.get("questions", []); file_level_goal = data.get("goal", "Unknown Goal From File")
    gen_logger.info(f"LOAD_PREP: Processing {len(raw_questions)} raw questions from {path} (File Goal: {file_level_goal}).")
    processed_count = 0; skipped_count = 0
    for idx, q_data in enumerate(raw_questions):
        q_copy = q_data.copy(); q_original_text_snippet = q_data.get("question", f"N/A_Q_IDX_{idx}")[:60]
        q_copy['goal_from_file'] = q_data.get('goal', file_level_goal)
        required_keys = ["type", "question", "answer", "difficulty", "topic"]
        missing_or_empty_keys = [k for k in required_keys if not q_copy.get(k)]
        if missing_or_empty_keys: skipped_count += 1; gen_logger.debug(f"LOAD_PREP: Skipping Q '{q_original_text_snippet}...' due to missing/empty: {missing_or_empty_keys}"); continue
        q_type_lower = q_copy["type"].lower()
        if q_type_lower == "mcq":
            options_input = q_copy.get("options"); answer_input = q_copy.get("answer") # answer_input is key if options is dict, text if options is list
            if not options_input or answer_input is None: skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping MCQ '{q_original_text_snippet}...' - missing options/answer."); continue
            
            temp_options_list = []
            temp_answer_text = ""

            if isinstance(options_input, dict):
                temp_options_list = [str(v).strip() for v in options_input.values() if str(v).strip()]
                correct_option_text_from_dict = options_input.get(str(answer_input)) # answer_input is key
                if correct_option_text_from_dict is None or not str(correct_option_text_from_dict).strip():
                    skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping MCQ '{q_original_text_snippet}...' - answer key '{answer_input}' not in options dict or answer text is empty."); continue
                temp_answer_text = str(correct_option_text_from_dict).strip()
            elif isinstance(options_input, list):
                temp_options_list = [str(opt).strip() for opt in options_input if str(opt).strip()]
                temp_answer_text = str(answer_input).strip() # answer_input is text
                if not temp_answer_text or temp_answer_text not in temp_options_list:
                    skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping MCQ '{q_original_text_snippet}...' - answer text '{temp_answer_text}' is empty or not in options list {temp_options_list}."); continue
            else:
                skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping MCQ '{q_original_text_snippet}...' - invalid options format (not dict or list)."); continue
            
            if not temp_options_list: # Ensure list is not empty after processing
                skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping MCQ '{q_original_text_snippet}...' - options_list became empty after processing."); continue

            q_copy["options_list"] = temp_options_list # This is List[str]
            q_copy["answer_text"] = temp_answer_text   # This is str (text of correct option)

        elif q_type_lower == "short_answer":
            q_copy["options_list"] = None # Explicitly None for SA
            q_copy["answer_text"] = str(q_copy.get("answer", "")).strip()
            if not q_copy["answer_text"]: skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping SA '{q_original_text_snippet}...' - empty answer."); continue
        else: skipped_count +=1; gen_logger.debug(f"LOAD_PREP: Skipping Q '{q_original_text_snippet}...' - unknown type."); continue
        
        prepared_questions.append(q_copy); processed_count +=1
    gen_logger.info(f"LOAD_PREP: Finished. Loaded {processed_count} questions, skipped {skipped_count}.")
    if not prepared_questions: gen_logger.warning("LOAD_PREP: No valid source questions loaded.")
    return prepared_questions
ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE = load_and_prepare_questions()


class RetrievalGenerator: # No changes needed here if it uses options_list and answer_text internally
    # ... (Keep the full logged version from previous answer)
    def __init__(self, all_questions_list: List[Dict[str, Any]], vectorizer_path: str = TFIDF_VECTORIZER_PATH_G):
        self.all_questions = all_questions_list; self.vectorizer: Optional[Any] = self._load_vectorizer(vectorizer_path)
        self.tfidf_matrix: Optional[Any] = None
        if self.vectorizer and self.all_questions:
            gen_logger.info(f"RetrievalGenerator: Building TF-IDF corpus from {len(self.all_questions)} questions...")
            self.corpus_texts: List[str] = [self._get_text_for_tfidf(q) for q in self.all_questions]
            try: self.tfidf_matrix = self.vectorizer.transform(self.corpus_texts); gen_logger.info(f"RetrievalGenerator: TF-IDF corpus built with shape {self.tfidf_matrix.shape}.")
            except Exception as e: gen_logger.error(f"RetrievalGenerator: Error building TF-IDF matrix: {e}", exc_info=True); self.tfidf_matrix = None
        elif not self.vectorizer: gen_logger.warning("RetrievalGenerator: TF-IDF vectorizer not loaded.")
        elif not self.all_questions: gen_logger.warning("RetrievalGenerator: No questions for TF-IDF corpus.")
    def _load_vectorizer(self, path: str) -> Optional[Any]: # ... (as before)
        if not os.path.exists(path): gen_logger.warning(f"RetrievalGenerator: TF-IDF vectorizer file not found at {path}."); return None
        try:
            with open(path, "rb") as f: vectorizer = pickle.load(f)
            gen_logger.info(f"RetrievalGenerator: TF-IDF vectorizer loaded successfully from {path}.")
            return vectorizer
        except Exception as e: gen_logger.error(f"RetrievalGenerator: Error loading TF-IDF vectorizer from {path}: {e}", exc_info=True); return None
    def _get_text_for_tfidf(self, question_data: Dict[str, Any]) -> str: # ... (as before)
        options_for_tfidf = question_data.get("options_list", [])
        if options_for_tfidf is None: options_for_tfidf = [] # Ensure it's a list for join
        text_parts = [question_data.get("question", ""), " ".join(options_for_tfidf), str(question_data.get("topic", "")), question_data.get("answer_text", ""), question_data.get("goal_from_file", "")]
        return " ".join(part for part in text_parts if part).lower()
    def get_questions(self, goal: str, num_to_retrieve: int, difficulty: str, topic: Optional[str] = None, q_type_filter: Optional[str] = None) -> List[Dict[str, Any]]: # ... (as before, with logging)
        if not self.all_questions: gen_logger.warning("Retrieval: No questions in bank."); return []
        gen_logger.debug(f"Retrieval: Filtering for goal='{goal}', diff='{difficulty}', topic='{topic}', type='{q_type_filter}' from {len(self.all_questions)} total questions.")
        candidates = [q for q in self.all_questions if q.get("goal_from_file", "").lower() == goal.lower() and q.get("difficulty", "").lower() == difficulty.lower()]
        if topic: candidates = [q for q in candidates if str(q.get("topic", "")).lower() == topic.lower()]
        if q_type_filter: candidates = [q for q in candidates if q.get("type", "").lower() == q_type_filter.lower()]
        gen_logger.debug(f"Retrieval: Found {len(candidates)} candidates after initial filtering.")
        if not candidates: gen_logger.info(f"Retrieval: No candidate questions found after all filters for goal='{goal}', diff='{difficulty}'."); return []
        ranked_questions = list(candidates); retrieval_method_detail = "retrieval_filtered_random_shuffle"
        if self.vectorizer and self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] == len(self.all_questions) and ranked_questions:
            query_parts = [goal, difficulty];
            if topic: query_parts.append(topic)
            query_text = " ".join(p for p in query_parts if p).lower()
            gen_logger.debug(f"Retrieval: Using TF-IDF. Query text: '{query_text}'")
            try:
                candidate_texts_for_ranking = [self._get_text_for_tfidf(c) for c in ranked_questions]
                if candidate_texts_for_ranking:
                    cand_matrix_for_ranking = self.vectorizer.transform(candidate_texts_for_ranking)
                    query_vector = self.vectorizer.transform([query_text])
                    similarities_for_ranking = cosine_similarity(cand_matrix_for_ranking, query_vector).flatten()
                    sorted_indices = similarities_for_ranking.argsort()[::-1]
                    ranked_questions = [ranked_questions[i] for i in sorted_indices]
                    retrieval_method_detail = "retrieval_tfidf"
                    gen_logger.debug(f"Retrieval: TF-IDF ranking applied.")
            except Exception as e_tfidf: gen_logger.error(f"Retrieval: Error during TF-IDF ranking: {e_tfidf}", exc_info=True); random.shuffle(ranked_questions)
        else: gen_logger.info("Retrieval: TF-IDF not used. Using random shuffle."); random.shuffle(ranked_questions)
        final_retrieved = []; seen_texts = set()
        for q_data in ranked_questions:
            if len(final_retrieved) >= num_to_retrieve: break
            q_text = q_data.get('question')
            if q_text and q_text.lower().strip() not in seen_texts:
                q_data["retrieval_method_detail"] = retrieval_method_detail 
                final_retrieved.append(q_data); seen_texts.add(q_text.lower().strip())
        gen_logger.info(f"Retrieval: Returning {len(final_retrieved)} questions for goal='{goal}'.")
        return final_retrieved


class QuizGeneratorT5: # ... (init, _extract_data_for_template, _generate_text_via_pipeline as before with logging)
    # ... (Keep the full logged version from previous answer for init, _extract_data, _generate_text_via_pipeline)
    def __init__(self, question_bank_for_seeding: List[Dict[str, Any]], templates_json_path: str = TEMPLATES_PATH_G): # ... (as before, with logging)
        self.question_bank_for_t5 = question_bank_for_seeding; self.templates: List[Dict[str,Any]] = []; self.text2text_pipeline: Optional[Any] = None
        if not os.path.exists(templates_json_path): gen_logger.error(f"T5Generator: Templates file not found: {templates_json_path}. T5 disabled."); return
        try:
            with open(templates_json_path, 'r', encoding='utf-8') as f: self.templates = json.load(f)
            if not self.templates: gen_logger.warning(f"T5Generator: Templates file empty: {templates_json_path}. T5 limited."); return
            gen_logger.info(f"T5Generator: Loaded {len(self.templates)} templates.")
        except Exception as e: gen_logger.error(f"T5Generator: Error loading templates: {e}", exc_info=True); return
        gen_logger.info(f"T5Generator: Initializing T5 model '{MODEL_NAME_QG_G}'...")
        try:
            self.qg_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME_QG_G); self.qg_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_QG_G)
            if self.qg_tokenizer.pad_token is None: self.qg_tokenizer.pad_token = self.qg_tokenizer.eos_token
            self.qg_model.config.pad_token_id = self.qg_tokenizer.pad_token_id
            self.text2text_pipeline = pipeline("text2text-generation", model=self.qg_model, tokenizer=self.qg_tokenizer, device=-1)
            gen_logger.info("T5Generator: T5 Model and pipeline initialized.")
        except Exception as e: gen_logger.critical(f"T5Generator: FATAL ERROR initializing HF components: {e}. T5 disabled.", exc_info=True); self.text2text_pipeline = None; self.templates = []
    def _extract_data_for_template(self, source_question: Dict[str, Any], template_def: Dict[str, Any]) -> Optional[Dict[str, Any]]: # ... (as before, with logging and NEED FOR IMPLEMENTATION)
        template_id = template_def.get("template_id", "unknown_template")
        gen_logger.debug(f"T5_EXTRACT: Attempting template '{template_id}' for source Q: '{source_question.get('question','')[:30]}...'")
        extracted_data = {"topic": source_question.get("topic"), "difficulty": source_question.get("difficulty"), "goal": source_question.get("goal_from_file"), "original_options_list": source_question.get("options_list", []), "original_question_text": source_question.get("question",""), "original_answer_text": source_question.get("answer_text","")}
        requirements = template_def.get("source_data_requirements", [])
        if template_id == "sa_define_concept_whatis": # ... (your logic)
            if source_question['type'].lower() != 'short_answer': gen_logger.debug(f"T5_EXTRACT_SKIP: '{template_id}' needs SA source."); return None
            concept_name = source_question.get('topic'); full_definition = source_question.get('answer_text')
            if not concept_name or not full_definition: gen_logger.debug(f"T5_EXTRACT_FAIL: '{template_id}' missing concept/definition."); return None
            extracted_data["concept_name"] = concept_name; extracted_data["full_definition"] = full_definition
        elif template_id in ["mcq_identify_command_for_function", "mcq_item_for_function"]: # ... (your logic)
            if source_question['type'].lower() != 'mcq': gen_logger.debug(f"T5_EXTRACT_SKIP: '{template_id}' needs MCQ source."); return None
            extracted_data["correct_item_name"] = source_question.get('answer_text') # This is now the TEXT of the correct answer
            q_text = source_question.get('question', '')
            match_func = re.search(r"(?:used to|command for|purpose is to|allows you to|function of|what does.*?do)\s+([\w\s,()'.\-:]+?)(?:[.?]|$|\s+in|\s+on)", q_text, re.IGNORECASE)
            if match_func and match_func.group(1).strip(): extracted_data["function_description"] = match_func.group(1).strip()
            else: extracted_data["function_description"] = f"perform a specific task related to {source_question.get('topic', 'the subject')}"
            if not extracted_data.get("correct_item_name"): gen_logger.debug(f"T5_EXTRACT_FAIL: '{template_id}' missing correct_item_name."); return None
        else: gen_logger.debug(f"T5_EXTRACT: No specific logic for template '{template_id}'.")
        for req_key in requirements:
            if req_key not in extracted_data or not extracted_data[req_key] or (isinstance(extracted_data[req_key], str) and not extracted_data[req_key].strip()):
                gen_logger.debug(f"T5_EXTRACT_FAIL: Template '{template_id}' - missing/empty required key '{req_key}'.")
                return None
        gen_logger.debug(f"T5_EXTRACT_SUCCESS: Template '{template_id}'.")
        return extracted_data
    def _generate_text_via_pipeline(self, prompt_text: str, **kwargs) -> str: # ... (as before, with logging and config for T5 params)
        if not self.text2text_pipeline: gen_logger.error("T5_PIPELINE: Not initialized."); return ""
        try:
            default_t5_params = {"max_length": 100, "min_length": 15, "num_beams":3, "temperature":0.85, "top_k":50, "top_p":0.95, "do_sample":True, "no_repeat_ngram_size":2}
            t5_params_from_config = APP_CONFIG_GENERATOR_SCOPE.get("t5_generation_params", default_t5_params)
            final_params = {**t5_params_from_config, **kwargs}
            gen_logger.debug(f"T5_PIPELINE_PARAMS: Using {final_params} for prompt: '{prompt_text[:70]}...'")
            pipeline_output = self.text2text_pipeline(prompt_text, **final_params)
            if pipeline_output and len(pipeline_output) > 0:
                generated_text = pipeline_output[0]['generated_text'].strip()
                generated_text = re.sub(r'\s+([?.!,"])', r'\1', generated_text)
                if generated_text and not re.search(r'[.?!]$', generated_text):
                    if "?" in prompt_text or "question" in prompt_text.lower(): generated_text += "?"
                    else: generated_text += "."
                return generated_text
            gen_logger.warning(f"T5_PIPELINE: Empty output for prompt: '{prompt_text[:70]}...'")
        except Exception as e_pipe: gen_logger.error(f"T5_PIPELINE: Error during generation: {e_pipe}", exc_info=True)
        return ""

    # MODIFIED: _generate_mcq_options_t5 to return List[str] for options and str for answer_text
    def _generate_mcq_options_t5(self, correct_answer_text_str: str, topic_str: str, original_options_list: List[str], num_options: int = 4) -> Tuple[Optional[List[str]], Optional[str]]:
        correct_answer_text_str = str(correct_answer_text_str).strip()
        if not correct_answer_text_str:
            gen_logger.error("T5_MCQ_OPTIONS: Correct answer text is empty. Cannot generate options.")
            return None, None # Return None for both if correct answer is invalid

        # Start with the correct answer. Use a set to manage uniqueness easily.
        current_options_set = {correct_answer_text_str}

        # Add unique distractors from the original options list
        distractors_from_source = [str(opt).strip() for opt in original_options_list if opt and str(opt).strip().lower() != correct_answer_text_str.lower()]
        random.shuffle(distractors_from_source)
        for distractor in distractors_from_source:
            if len(current_options_set) >= num_options: break
            if distractor: current_options_set.add(distractor)
        
        final_mcq_options_list = list(current_options_set)
        placeholders_added_count = 0; option_idx = 0
        while len(final_mcq_options_list) < num_options:
            placeholders_added_count +=1
            placeholder_opt = f"Placeholder Option {chr(ord('P') + option_idx)}"
            if placeholder_opt not in final_mcq_options_list: final_mcq_options_list.append(placeholder_opt)
            else: final_mcq_options_list.append(f"{placeholder_opt} ({random.randint(100,999)})")
            option_idx += 1
            if option_idx > (num_options * 2): # Safety break
                gen_logger.error(f"T5_MCQ_OPTIONS: Excessive placeholder generation for '{correct_answer_text_str}'.")
                break 
        
        if placeholders_added_count > 0:
            gen_logger.warning(f"T5_MCQ_OPTIONS: Used {placeholders_added_count} placeholder(s) for options for answer '{correct_answer_text_str}' on topic '{topic_str}'.")

        # Ensure correct number of options
        if len(final_mcq_options_list) > num_options: final_mcq_options_list = final_mcq_options_list[:num_options]
        elif len(final_mcq_options_list) < num_options: # Should be rare with padding
            gen_logger.error(f"T5_MCQ_OPTIONS: Failed to generate {num_options} options for '{correct_answer_text_str}'. Got {len(final_mcq_options_list)}.")
            while len(final_mcq_options_list) < num_options: final_mcq_options_list.append(f"ErrorOption{len(final_mcq_options_list)+1}")
        
        random.shuffle(final_mcq_options_list)

        # Ensure the correct answer is still in the list (it should be)
        if correct_answer_text_str not in final_mcq_options_list:
            gen_logger.error(f"T5_MCQ_OPTIONS: CRITICAL - Correct answer '{correct_answer_text_str}' was lost. Re-adding and shuffling.")
            if final_mcq_options_list: final_mcq_options_list[0] = correct_answer_text_str # Replace first
            else: final_mcq_options_list.append(correct_answer_text_str) # Should not happen
            random.shuffle(final_mcq_options_list)

        return final_mcq_options_list, correct_answer_text_str

    def generate_t5_questions(self, goal_str: str, num_questions_to_gen: int, difficulty_pref: str, topic_pref: Optional[str] = None, q_type_pref: Optional[str] = None) -> List[Dict[str, Any]]:
        # ... (seed candidate and template filtering as before, with logging) ...
        generated_quiz_items = []
        if not self.templates or not self.text2text_pipeline: gen_logger.warning("T5_GEN: Cannot generate. No templates or T5 pipeline."); return []
        if not self.question_bank_for_t5: gen_logger.warning("T5_GEN: Cannot generate. No seed question bank."); return []
        seed_candidates = [q for q in self.question_bank_for_t5 if q.get('goal_from_file', '').lower() == goal_str.lower() and q.get('difficulty', '').lower() == difficulty_pref.lower()]
        if topic_pref: seed_candidates = [q for q in seed_candidates if str(q.get('topic', "")).lower() == topic_pref.lower()]
        gen_logger.info(f"T5_GEN: Found {len(seed_candidates)} seed candidates for goal='{goal_str}', diff='{difficulty_pref}', topic_pref='{topic_pref}'.")
        if not seed_candidates: gen_logger.warning("T5_GEN: No suitable seed questions found."); return []
        applicable_templates = list(self.templates)
        if q_type_pref: applicable_templates = [t for t in applicable_templates if t.get("question_type", "").lower() == q_type_pref.lower()]
        gen_logger.info(f"T5_GEN: Found {len(applicable_templates)} applicable templates for q_type_pref='{q_type_pref}'.")
        if not applicable_templates: gen_logger.warning("T5_GEN: No applicable templates found."); return []
        attempts = 0; max_attempts_factor = APP_CONFIG_GENERATOR_SCOPE.get("t5_max_attempts_factor", 10); max_total_attempts = num_questions_to_gen * len(applicable_templates) * max_attempts_factor; generated_q_texts = set()
        gen_logger.info(f"T5_GEN: Attempting to generate {num_questions_to_gen} questions. Max attempts: {max_total_attempts}.")

        while len(generated_quiz_items) < num_questions_to_gen and attempts < max_total_attempts:
            # ... (template/seed selection, data extraction, prompt generation, T5 call, QC as before) ...
            attempts += 1; selected_template = random.choice(applicable_templates); source_q_seed = random.choice(seed_candidates)
            template_id = selected_template.get("template_id", "unknown"); template_q_type = selected_template.get("question_type","").lower()
            source_q_type_lower = source_q_seed.get("type","").lower()
            if (template_q_type == "mcq" and source_q_type_lower != "mcq") or (template_q_type == "short_answer" and source_q_type_lower != "short_answer"): gen_logger.debug(f"T5_GEN_SKIP (Attempt {attempts}): Template '{template_id}' type mismatch source Q type."); continue
            extracted = self._extract_data_for_template(source_q_seed, selected_template)
            if not extracted: gen_logger.debug(f"T5_GEN_SKIP (Attempt {attempts}): Data extraction failed for template '{template_id}'."); continue
            try: prompt = selected_template["t5_prompt_structure"].format(**extracted)
            except KeyError as e: gen_logger.error(f"T5_GEN_ERROR (Attempt {attempts}): Prompt format KeyError for '{template_id}': {e}.", exc_info=True); continue
            gen_logger.debug(f"T5_GEN_PROMPT (Attempt {attempts}, Template '{template_id}'): {prompt[:150]}...")
            new_q_text = self._generate_text_via_pipeline(prompt)
            gen_logger.debug(f"T5_GEN_RAW_OUTPUT (Attempt {attempts}): '{new_q_text[:100]}...'")
            original_q_text_lower = source_q_seed.get("question","").lower()
            if not new_q_text or len(new_q_text.strip()) < 20 or any(kw.lower() in new_q_text.lower() for kw in ["generate a question", "context:", "based on the information", "given the following", "placeholder", "your task is to", "example:", "note:"]) or new_q_text.lower().strip() == prompt.lower().strip() or (original_q_text_lower and len(original_q_text_lower)>10 and original_q_text_lower in new_q_text.lower()):
                gen_logger.debug(f"T5_GEN_QC_FAIL (Attempt {attempts}): Output bad. Output: '{new_q_text[:70]}...'"); continue
            norm_new_q_text = ' '.join(new_q_text.lower().split())
            if norm_new_q_text in generated_q_texts: gen_logger.debug(f"T5_GEN_DEDUPE (Attempt {attempts}): Skipping duplicate text."); continue
            generated_q_texts.add(norm_new_q_text)

            answer_slot_key = selected_template.get("answer_slot")
            correct_answer_text_from_slot = extracted.get(answer_slot_key) # This is the TEXT of the answer
            if correct_answer_text_from_slot is None:
                gen_logger.error(f"T5_GEN_ERROR (Attempt {attempts}): Answer not found via slot '{answer_slot_key}' for template '{template_id}'.")
                continue

            item: Dict[str, Any] = {
                "type": template_q_type, "question": new_q_text.strip(),
                "difficulty": difficulty_pref, "topic": extracted.get("topic"),
                "goal": goal_str, # This will be filtered by main.py if not in Pydantic QuestionResponse
                "generation_method": "t5_generated", "t5_template_id": template_id # Internal fields
            }

            if template_q_type == "mcq":
                mcq_options_list, mcq_answer_text_final = self._generate_mcq_options_t5(
                    str(correct_answer_text_from_slot),
                    extracted.get("topic", "general topic"),
                    extracted.get("original_options_list", []) # Seed with original options
                )
                if not mcq_options_list or mcq_answer_text_final is None:
                    gen_logger.error(f"T5_GEN_ERROR (Attempt {attempts}): MCQ options/answer_text generation failed for template '{template_id}'.")
                    continue
                item["options"] = mcq_options_list # List[str]
                item["answer"] = mcq_answer_text_final   # str (text of correct option)
            else: # short_answer
                item["answer"] = str(correct_answer_text_from_slot)
                item["options"] = None # Explicitly None for SA

            generated_quiz_items.append(item)
            gen_logger.info(f"T5_GEN_SUCCESS: Generated Q#{len(generated_quiz_items)} via '{template_id}'.")
        
        if len(generated_quiz_items) < num_questions_to_gen: gen_logger.warning(f"T5_GEN: Only generated {len(generated_quiz_items)}/{num_questions_to_gen} T5 questions after {attempts} attempts.")
        else: gen_logger.info(f"T5_GEN: Successfully generated {len(generated_quiz_items)} T5 questions.")
        return generated_quiz_items


class QuizMaster:
    def __init__(self, app_config_from_main: Dict[str, Any]): # ... (as before)
        global APP_CONFIG_GENERATOR_SCOPE; APP_CONFIG_GENERATOR_SCOPE = app_config_from_main
        gen_logger.info("QuizMaster initializing components...")
        self.retrieval_generator = RetrievalGenerator(ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE, TFIDF_VECTORIZER_PATH_G)
        self.t5_generator_instance: Optional[QuizGeneratorT5] = QuizGeneratorT5(ALL_PREPARED_QUESTIONS_GENERATOR_SCOPE, TEMPLATES_PATH_G)
        if self.t5_generator_instance and (not self.t5_generator_instance.templates or not self.t5_generator_instance.text2text_pipeline):
            gen_logger.warning("QuizMaster: T5 Generator was not fully initialized. T5 generation will be skipped.")
            self.t5_generator_instance = None
        gen_logger.info("QuizMaster components initialized.")

    # MODIFIED: _format_retrieved_question to output options as List[str] and answer as str (text)
    def _format_retrieved_question(self, q_from_bank: Dict[str, Any], target_goal_for_response: str) -> Optional[Dict[str, Any]]:
        # q_from_bank should have 'options_list' (List[str]) and 'answer_text' (str)
        # from load_and_prepare_questions.
        formatted_q_dict = {
            "type": q_from_bank.get("type","").lower(),
            "question": q_from_bank.get("question"),
            "difficulty": q_from_bank.get("difficulty"),
            "topic": q_from_bank.get("topic"),
            # "goal": target_goal_for_response, # 'goal' is no longer in QuestionResponse Pydantic model
            # Internal field, will be filtered by main.py if not in Pydantic QuestionResponse
            "generation_method": q_from_bank.get("retrieval_method_detail", "retrieval_bank_unknown_detail")
        }
        q_type = formatted_q_dict["type"]

        if q_type == "mcq":
            options_list_from_bank = q_from_bank.get("options_list") # Expected List[str]
            correct_answer_text_from_bank = q_from_bank.get("answer_text") # Expected str

            if not isinstance(options_list_from_bank, list) or not options_list_from_bank or \
               correct_answer_text_from_bank is None or not str(correct_answer_text_from_bank).strip():
                gen_logger.warning(f"QM_FORMAT_MCQ: Bad data for retrieved MCQ '{q_from_bank.get('question','')[:30]}...'. Invalid options_list or answer_text. Skipping.")
                return None
            
            current_options = list(options_list_from_bank) # Make a mutable copy
            placeholders_added_count = 0; idx = 0
            while len(current_options) < 4:
                placeholders_added_count += 1
                placeholder = f"Option {chr(ord('X')+idx)}"
                if placeholder not in current_options: current_options.append(placeholder)
                else: current_options.append(f"AltOpt {chr(ord('X')+idx)}{random.randint(1,9)}")
                idx+=1
            
            if placeholders_added_count > 0:
                gen_logger.warning(f"QM_FORMAT_MCQ: Added {placeholders_added_count} placeholder(s) to options for Q: '{q_from_bank.get('question','')[:30]}...'")

            # Ensure correct answer is in the list, then shuffle
            str_correct_answer_text = str(correct_answer_text_from_bank).strip()
            if str_correct_answer_text not in current_options:
                gen_logger.warning(f"QM_FORMAT_MCQ: Correct answer '{str_correct_answer_text}' not in options list {current_options}. Adding it.")
                current_options.append(str_correct_answer_text)
                if len(current_options) > 4 : # If over 4 options now, trim intelligently
                    # Try to remove a placeholder if one was added
                    placeholder_removed = False
                    for i, opt_text in reversed(list(enumerate(current_options))): # Iterate backwards to safely pop
                        if "Placeholder Option" in opt_text or "AltOpt" in opt_text or "Option X" in opt_text : # General placeholder check
                            if opt_text != str_correct_answer_text : # Don't remove the answer if it was a placeholder
                                current_options.pop(i)
                                placeholder_removed = True
                                break
                    if not placeholder_removed and len(current_options) > 4: # If no placeholder removed, remove last (unless it's the answer)
                        if current_options[-1] != str_correct_answer_text: current_options.pop()
                        else: current_options.pop(0) # Remove first if last is answer
            
            random.shuffle(current_options)
            final_options_list = current_options[:4] if len(current_options) >=4 else current_options
            # Final check to ensure answer is present if list was trimmed
            if str_correct_answer_text not in final_options_list and final_options_list:
                final_options_list[0] = str_correct_answer_text
                random.shuffle(final_options_list)


            formatted_q_dict["options"] = final_options_list # List[str]
            formatted_q_dict["answer"] = str_correct_answer_text # str (text of correct option)

        elif q_type == "short_answer":
            formatted_q_dict["answer"] = q_from_bank.get("answer_text")
            formatted_q_dict["options"] = None # Explicitly None for SA
        else:
            gen_logger.warning(f"QM_FORMAT: Retrieved question with unknown type '{q_type}'. Cannot format.")
            return None
        return formatted_q_dict

    def generate_quiz_hybrid(self, goal: str, num_questions_total: int, difficulty: str,
                             topic: Optional[str] = None, q_type: Optional[str] = None) -> List[Dict[str, Any]]:
        # ... (Logic to determine effective_mode from APP_CONFIG_GENERATOR_SCOPE as before) ...
        # ... (Phases for retrieval and T5 as before, ensuring they call methods that produce the new format) ...
        final_questions: List[Dict[str, Any]] = []; seen_q_texts = set()
        effective_mode = APP_CONFIG_GENERATOR_SCOPE.get("generator_mode", "hybrid").lower()
        gen_logger.info(f"QuizMaster: Using configured mode '{effective_mode}' for goal='{goal}', num_req={num_questions_total}, diff='{difficulty}', topic='{topic}', type='{q_type}'")
        num_ret_needed, num_t5_needed = 0, 0
        if effective_mode in ["retrieval", "retrieval_only"]: num_ret_needed = num_questions_total; gen_logger.info(f"QM: Mode '{effective_mode}', will attempt {num_ret_needed} from retrieval.")
        elif effective_mode in ["t5_only", "template"]: num_t5_needed = num_questions_total; gen_logger.info(f"QM: Mode '{effective_mode}', will attempt {num_t5_needed} from T5/template.")
        elif effective_mode == "hybrid": num_ret_needed = num_questions_total; gen_logger.info(f"QM: Mode '{effective_mode}', will attempt {num_ret_needed} from retrieval first, then T5 fallback.")
        else: gen_logger.warning(f"QM: Unknown configured generator_mode '{effective_mode}', defaulting to hybrid."); num_ret_needed = num_questions_total; effective_mode = "hybrid"
        if num_ret_needed > 0 and self.retrieval_generator:
            gen_logger.info(f"QM: Retrieval phase starting for {num_ret_needed} questions.")
            retrieved_raw = self.retrieval_generator.get_questions(goal, num_ret_needed, difficulty, topic, q_type)
            count_added_retrieval = 0
            for q_raw in retrieved_raw:
                if len(final_questions) >= num_questions_total : break
                q_text_norm = q_raw.get("question","").lower().strip()
                if q_text_norm and q_text_norm not in seen_q_texts:
                    formatted = self._format_retrieved_question(q_raw, goal) # This now returns new format
                    if formatted: final_questions.append(formatted); seen_q_texts.add(q_text_norm); count_added_retrieval+=1
            gen_logger.info(f"QM: Retrieval phase added {count_added_retrieval} unique questions. Total now: {len(final_questions)}.")
        if effective_mode == "hybrid":
            num_t5_needed = num_questions_total - len(final_questions)
            if num_t5_needed > 0: gen_logger.info(f"QM: Hybrid mode: {num_t5_needed} more questions needed from T5.")
            elif len(final_questions) >= num_questions_total: gen_logger.info("QM: Hybrid mode: Retrieval fulfilled all question needs."); num_t5_needed = 0
        if num_t5_needed > 0:
            if self.t5_generator_instance:
                gen_logger.info(f"QM: T5/Template phase starting for {num_t5_needed} questions.")
                t5_generated = self.t5_generator_instance.generate_t5_questions(goal, num_t5_needed, difficulty, topic, q_type) # This now returns new format
                count_added_t5 = 0
                for q_t5 in t5_generated:
                    if len(final_questions) >= num_questions_total: break
                    q_text_norm = q_t5.get("question","").lower().strip()
                    if q_text_norm and q_text_norm not in seen_q_texts:
                        final_questions.append(q_t5); seen_q_texts.add(q_text_norm); count_added_t5+=1
                gen_logger.info(f"QM: T5/Template phase added {count_added_t5} unique questions. Total now: {len(final_questions)}")
            else: gen_logger.warning(f"QM: T5 instance not available. Could not generate {num_t5_needed} T5 questions.")
        gen_logger.info(f"QuizMaster finished. Returning {len(final_questions[:num_questions_total])} of {num_questions_total} requested questions.")
        return final_questions[:num_questions_total]