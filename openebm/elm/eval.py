import json
from openebm.elm import metrics
# import nltk

# try:
#     nltk.data.find("tokenizers/punkt_tab")
# except LookupError:
#     nltk.download('punkt_tab')
    
    
def nlp_eval_acc(answer_file):
    data_name = answer_file.split("/")[-1].split("_")[-1].split(".")[0]
    model_name = '_'.join(answer_file.split("/")[-1].split("_")[:-1])
    predictions = []
    references = []

    with open(answer_file, 'r', encoding='utf-8') as f:
        skipped = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            output = data['generation']
            # Support both old and new field names for backward compatibility
            # New: 'target', Old: 'gt_answer'
            gt_ans = data.get('target', data.get('gt_answer', '')).split("\n####")[-1].strip()

            predictions.append(output)
            references.append(gt_ans)
        if skipped > 0:
            print(f"WARNING: Skipped {skipped} corrupted JSON lines in {answer_file} (possibly caused by concurrent writes)")

    em_score = metrics.calculate_em(predictions, references)
    f1_score = metrics.calculate_f1_score(predictions, references)

    print(f"{model_name} Results on {data_name}:")
    print(f"Exact Match Accuracy: {100*em_score:.3f}%")
    print(f"F1 Score: {100*f1_score:.3f}%")
    return em_score, f1_score
