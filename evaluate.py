from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import numpy as np

def read_answers(file_path):
    """
    Read answers from a text file, splitting answers by empty lines.
    :param file_path: Path to the text file.
    :return: A list of answers.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return [answer.strip() for answer in content.split('\n\n') if answer.strip()]

def evaluate_answers(model_answers, reference_answers):
    """
    Evaluate the similarity between model answers and reference answers using various metrics.
    :param model_answers: List of model-generated answers.
    :param reference_answers: List of reference answers.
    :return: A dictionary containing evaluation results.
    """
    if len(model_answers) != len(reference_answers):
        raise ValueError("The number of model answers and reference answers must match.")
    
    # Initialize evaluation metrics
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    cosine_similarities = []
    exact_matches = []
    fuzzy_matches = []

    # Initialize NLP models
    rouge = Rouge()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('E:\project\paraphrase-MiniLM-L6-v2')


    # Compare each pair of model answer and reference answer
    for model_ans, ref_ans in zip(model_answers, reference_answers):
        # BLEU score
        bleu = sentence_bleu([ref_ans.split()], model_ans.split())
        bleu_scores.append(bleu)

        # ROUGE scores
        rouge_score = rouge.get_scores(model_ans, ref_ans)[0]
        for key in rouge_scores:
            rouge_scores[key].append(rouge_score[key]['f'])

        # Cosine Similarity
        embeddings = model.encode([model_ans, ref_ans])
        cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        cosine_similarities.append(cosine_sim)

        # Exact Match
        exact_matches.append(int(model_ans.strip() == ref_ans.strip()))

        # Fuzzy Match
        fuzzy_matches.append(fuzz.ratio(model_ans, ref_ans))

    # Calculate average scores
    results = {
        'Cosine Similarity': np.mean(cosine_similarities),
        'Fuzzy Match': np.mean(fuzzy_matches),
    }
    return results

# File paths
model_file_path = "C:\\Users\\13157\\Desktop\\answer.txt"
reference_file_path = "C:\\Users\\13157\\Desktop\\referenceanswer.txt"

# Read answers
model_answers = read_answers(model_file_path)
reference_answers = read_answers(reference_file_path)

# Evaluate answers
try:
    evaluation_results = evaluate_answers(model_answers, reference_answers)
    print("Evaluation Results:")
    for metric, score in evaluation_results.items():
        print(f"{metric}: {score:.4f}")
except ValueError as e:
    print(f"Error: {e}")
