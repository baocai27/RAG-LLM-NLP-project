import os
import jieba
import pickle
import pdfplumber
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

device = "cuda"

rerank_tokenizer = AutoTokenizer.from_pretrained(r'E:\project\bge-reranker-base')
rerank_model = AutoModelForSequenceClassification.from_pretrained(r'E:\project\bge-reranker-base')
rerank_model.cuda()

model_path = r'E:\project\Qwen2_7B_Instruct_AWQ'
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype="auto",
    device_map="auto",
)

# Cache file paths
bm25_cache_path = "c:\\Users\\13157\\Desktop\\bm25_cache.pkl"
semantic_cache_path = "c:\\Users\\13157\\Desktop\\semantic_cache.pkl"


def save_cache(data, path):
    with open(path, "wb") as f:
        print(f"Saving cache to {path}")
        pickle.dump(data, f)


def load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except EOFError:
                print(f"Cache file {path} is empty, regenerating cache.")
                return None
    return None


def split_text_fixed_size(text, chunk_size, overlap_size):
    new_text = []
    for i in range(0, len(text), chunk_size):
        if i == 0:
            new_text.append(text[0:chunk_size])
        else:
            new_text.append(text[i - overlap_size:i + chunk_size])
    return new_text


def get_rank_index(max_score_page_idxs_, questions_, pdf_content_):
    pairs = []
    for idx in max_score_page_idxs_:
        pairs.append([questions_[query_idx]["question"], pdf_content_[idx]['content']])

    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    max_score = scores.cpu().numpy().argmax()
    index = max_score_page_idxs_[max_score]

    return max_score, index


def read_data(query_data_path, knowledge_data_paths):
    with open(query_data_path, 'r', encoding='utf-8') as f:
        questions = [{'question': line.strip()} for line in f if line.strip()]

    pdf_content = []
    for knowledge_data_path in knowledge_data_paths:
        pdf = pdfplumber.open(knowledge_data_path)
        for page_idx in range(len(pdf.pages)):
            text = pdf.pages[page_idx].extract_text()
            new_text = split_text_fixed_size(text, chunk_size=100, overlap_size=5)
            for chunk_text in new_text:
                pdf_content.append({
                    'page': f'{knowledge_data_path}_page_{page_idx + 1}',
                    'content': chunk_text
                })
    return questions, pdf_content


def qwen_preprocess(tokenizer_, ziliao, question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"帮我结合给定的资料，回答问题。如果问题答案无法从资料中获得，"
                                    f"输出结合给定的资料，无法回答问题. 如果找到答案, 就输出找到的答案, 资料：{ziliao}, 问题:{question}"},
    ]
    text = tokenizer_.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs_ = tokenizer_([text], return_tensors="pt").to(device)

    input_ids = tokenizer_.encode(text, return_tensors='pt')
    attention_mask_ = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    return model_inputs_, attention_mask_


if __name__ == '__main__':

    knowledge_data_paths = [
        r'c:\Users\13157\Desktop\documents\M1-航空概论R1.pdf',
        r'C:\Users\13157\Desktop\documents\M2-航空器维修R1.pdf',
        r'C:\Users\13157\Desktop\documents\M3-飞机结构和系统R1.pdf',
        r'C:\Users\13157\Desktop\documents\M4-直升机结构和系统.pdf',
        r'C:\Users\13157\Desktop\documents\M5-航空涡轮发动机R1.pdf',
        r'C:\Users\13157\Desktop\documents\M6-活塞发动机及其维修.pdf',
        r'C:\Users\13157\Desktop\documents\M7-航空器维修基本技能.pdf',
        r'C:\Users\13157\Desktop\documents\M8-航空器维修实践R1.pdf'
    ]

    questions, pdf_content = read_data(query_data_path=r"C:\Users\13157\Desktop\question.txt",
                                       knowledge_data_paths=knowledge_data_paths)

    # Try loading cache
    bm25 = load_cache(bm25_cache_path)
    semantic_data = load_cache(semantic_cache_path)

    if bm25 is None or semantic_data is None:
        print("Cache files are empty or missing, regenerating caches...")
        pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
        bm25 = BM25Okapi(pdf_content_words)

        sent_model = SentenceTransformer(r'E:\project\stella_base_zh_v3_1792d')
        question_sentences = [x['question'] for x in questions]
        pdf_content_sentences = [x['content'] for x in pdf_content]

        question_embeddings = sent_model.encode(question_sentences, normalize_embeddings=True)
        pdf_embeddings = sent_model.encode(pdf_content_sentences, normalize_embeddings=True)

        # Save BM25 and semantic data to cache
        save_cache(bm25, bm25_cache_path)
        save_cache((question_embeddings, pdf_embeddings), semantic_cache_path)
    else:
        print("Loaded cache from files.")
        question_embeddings, pdf_embeddings = semantic_data

    for query_idx in range(len(questions)):
        doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
        bm25_score_page_idxs = doc_scores.argsort()[-10:]

        score = question_embeddings[query_idx] @ pdf_embeddings.T
        ste_score_page_idxs = score.argsort()[-10:]

        bm25_score, bm25_index = get_rank_index(bm25_score_page_idxs, questions, pdf_content)
        ste_score, ste_index = get_rank_index(ste_score_page_idxs, questions, pdf_content)

        max_score_page_idx = ste_index if ste_score >= bm25_score else bm25_index

        model_inputs, attention_mask = qwen_preprocess(
            tokenizer, pdf_content[max_score_page_idx]['content'], questions[query_idx]["question"]
        )

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'question: {questions[query_idx]["question"]}, answer: {response}')
