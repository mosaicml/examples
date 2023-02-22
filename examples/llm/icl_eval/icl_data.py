from lm_eval import tasks as lm_eval_tasks
import json


def prep_lambada_row(row):
    ctxt, continuation = ' '.join(row['text'].split(' ')[:-1]), row['text'].split(' ')[-1]
    return {
        "context": ctxt, "continuation": continuation
    }

def prep_piqa_row(row):
    row['query'] = row['goal']
    del row['goal']
    return row


PREP = {
    'lambada_standard': prep_lambada_row,
    'lambada_openai': prep_lambada_row,
    'piqa': prep_piqa_row,
    'hellaswag': lambda x : x
}
if __name__ == "__main__":
    for task_name in ['lambada_standard', 'lambada_openai', 'piqa', 'hellaswag']:
        task = lm_eval_tasks.get_task_dict([task_name])[task_name]

        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs

        with open(f"{task_name}.jsonl", "w") as f:
            for row in task_doc_func():
                row = PREP[task_name](row)
                f.write(json.dumps(row) + '\n')
