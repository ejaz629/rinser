from datasets import *
from transformers import *
from tokenizers import *
import os
import json
import ast
import torch


model = BertForMaskedLM.from_pretrained(os.path.join(r'rinser-llm/checkpoint-28700'))
tokenizer = BertTokenizerFast.from_pretrained(r'rinser-llm')
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

apilist = []
count = 0
cor = 0
cor_context = 0
tot_count = 0


def predict_api(test_api, fill_mask):
    corr = False
    contxt_res = False
    valid_input = True
    actual = test_api[0]
    predicted = None
    example = "[MASK] " + ' '.join(test_api[1:])
    try:
        out = fill_mask(example)
    except RuntimeError:
        valid_input = False
        return actual, predicted, corr, contxt_res, valid_input

    for prediction in out:
        predicted = prediction["token_str"]
        if actual == predicted:
            corr = True
        break
    return actual, predicted, corr, contxt_res, valid_input



with open(r'api-codeprints-for-llm/test_apicodeprints_for_llm.txt', "r", encoding="utf8") as f:
    for line in f:
        corr = False
        line = line.strip()
        t_ex = line.split(" ")
        actual = t_ex[0]
        tot_count += 1
        if len(t_ex) <= 600 and len(t_ex) > 1:
            test_api = t_ex
            actual, predicted, corr, context_res, valid_input = predict_api(test_api, fill_mask)
            try:
                contx = ' '.join(t_ex[1:])
            except:
                contx = "not catched"
                pass
                    
            
            print(count)
            print("Actual: ", {actual})
            print("Predicted: ",{predicted})
            #print("API context: ",{contx})
            try:
                print("Prediction accuracy (%): ", cor / count)
            except:
                pass
            print("Unique APIs: ", len(set(apilist)))
            print("================================================================================================================")
            if not valid_input:
                continue
            if corr:
                apilist.append(actual)
                cor += 1
            if context_res:
                apilist.append(actual)
                cor_context += 1
            count += 1
        #if tot_count == 10:
         #   break

print("Total instances: ", tot_count)
print("Test API codeprints: ", count)
print("Correct predictions: ", cor)
print("Prediction accuracy (%): ", cor / count)
print("Total unique APIs predicted correctly: ", len(set(apilist)))

