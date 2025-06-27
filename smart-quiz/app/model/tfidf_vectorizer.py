import os
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
def load_questions():
    script_dir=os.path.dirname(__file__)
    path=os.path.join(script_dir,"..","..","data","question_bank.json")
    with open(path,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data.get("questions",[])
def build_corpus(questions):
    corpus=[]
    for q in questions:
        text=q.get("question","")
        options=q.get("options",{})
        if isinstance(options,dict):
            text+=" "+" ".join(options.values())
        elif isinstance(options,list):
            text+=" "+" ".join(options)
        corpus.append(text)
    return corpus
# tfidf_vectorizer.py
# ... (imports) ...
def main():
    questions=load_questions() # Path within load_questions() should be correct relative to this script
    corpus=build_corpus(questions)
    vectorizer=TfidfVectorizer()
    vectorizer.fit(corpus)

    # CHANGE THIS PART to save in the correct location for `generator.py`
    # Assuming tfidf_vectorizer.py is in the project root or `smart-quiz/`
    # and main.py/generator.py are in `smart-quiz/app/`
    script_dir=os.path.dirname(__file__) # Directory of tfidf_vectorizer.py
    # Path for generator.py to find the model
    model_dir = os.path.join(script_dir, "app", "model") # Create app/model if it doesn't exist
    model_path=os.path.join(model_dir, "tfidf_vectorizer.pkl")

    os.makedirs(os.path.dirname(model_path),exist_ok=True)
    with open(model_path,"wb") as f:
        pickle.dump(vectorizer,f)
    print(f"TF-IDF Vectorizer saved to: {model_path}")

if __name__ == "__main__":
    main()
