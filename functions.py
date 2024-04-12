import re
import pandas as pd
from tqdm import tqdm
import os
from transformers import *
from tokenizers import *
from numpy import dot
import numpy as np
from numpy.linalg import norm
import json
import warnings
import json
import os
import re
import string
import nltk
import math

# nltk.download('punkt')
from nltk.corpus import stopwords
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

regs = ['eax', 'esp', 'ecx', 'ebp', 'ebx', 'edx', 'esi', 'edi']


def append_record(record):
    with open('api-codeprints/fine_grained_api_contexts', 'a', newline="") as f:
        json.dump(record, f)
        f.write(os.linesep)


def identify_parameter_type(param, regs):
    p_type = 0
    if param.isnumeric():
        p_type = 1

    regex = r"\[(.*?)\]"
    matches = re.finditer(regex, param, re.MULTILINE)
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(), end=match.end(), match=match.group()))
        p_type = 4

    if param in regs:
        p_type = 3

    return p_type


def extract_reg(var, regs):
    reg = None
    regex = r"\[(.*?)\]"
    matches = re.finditer(regex, var, re.MULTILINE)
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(), end=match.end(), match=match.group()))

        parse = ''.join(re.findall(r"\[(.*?)\]", match.group()))
        items = parse.split("+")
        for item in items:
            if item in regs:
                reg = item
                break
    return reg


def process_instrs_set(instr_set):
    for instr in instr_set:
        ins = re.sub('\s+', ', ', instr, 1)
        parts = ins.split(', ')
        mnem = parts[1:]
        regex = r"\[(.*?)\]"
        matches = re.finditer(regex, ' '.join(mnem), re.MULTILINE)
        for matchNum, match in enumerate(matches):
            matchNum = matchNum + 1
            print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(),
                                                                                end=match.end(), match=match.group()))


def ishex(s):
    return not re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", s)


def get_operands(triggers, inst):
    ins_ = re.sub('\s+', ', ', inst, 1)
    parts_ = ins_.split(', ')
    for trigger in triggers:
        # print([triggers,inst, parts_[1:], trigger in parts_[1:]])
        if trigger in ' '.join(parts_[1:]):
            if len(parts_) == 2:
                trigger = parts_[len(parts_) - 1].split(';')[:(len(parts_) - 1)]
                return trigger
            elif len(parts_) > 2:
                operands = parts_[1:]
                return operands


def check_search_status(triggers, par):
    status = False
    for trigger in triggers:
        if trigger in par:
            status = True
            break
    return status


def track_parameters(sam, fnc_name, api, instr_set):
    p_fingerprints = []
    api_det = {}
    api_det['sample'] = sam
    api_det['fun_name'] = fnc_name
    api_det['api_name'] = api
    idx = 0
    for row in instr_set:
        ins = re.sub('\s+', ', ', row, 1)
        parts = ins.split(', ')
        mnem = parts[0]
        idx += 1
        if mnem == 'push' and ';' in parts[len(parts) - 1]:
            operand = ''.join(parts[len(parts) - 1].split(';')[:(len(parts) - 1)])
            annotations = ''.join(parts[len(parts) - 1].split(';')[-1].strip())
            triggers = []
            code = []
            param = {}
            if operand in regs:
                code.append(row)
                triggers.append(operand)
                for par in reversed(instr_set[:idx - 1]):
                    if 'call' in par and 'sub_' in par:
                        code.append(par)
                    if check_search_status(triggers, par):
                        vars = get_operands(triggers, par)
                        if vars:
                            for var in vars:
                                p_type = identify_parameter_type(var, regs)
                                if p_type == 3:
                                    if par not in code:
                                        code.append(par)
                                    if var not in triggers:
                                        triggers.append(var)
                                if p_type == 4:
                                    reg = extract_reg(var, regs)
                                    if reg:
                                        if par not in code:
                                            code.append(par)
                                        if reg not in triggers:
                                            triggers.append(reg)

                param['p_annot'] = annotations
                param['p_val'] = operand
                param['tracked_code'] = code
            else:
                param['p_annot'] = annotations
                param['p_val'] = operand
                param['tracked_code'] = None
            p_fingerprints.append(param)

    api_det["params"] = p_fingerprints
    return api_det


def track_parameters_for_stripped_binaries(sam, fnc_name, api, instr_set):
    p_fingerprints = []
    api_det = {}
    api_det['sample'] = sam
    api_det['fun_name'] = fnc_name
    api_det['api_name'] = api
    idx = 0
    for row in instr_set:
        ins = re.sub('\s+', ', ', row, 1)
        parts = ins.split(', ')
        mnem = parts[0]
        idx += 1
        if mnem == 'push':  # and ';' in parts[len(parts) - 1]:
            if ';' in parts[1]:
                operand = ''.join(parts[len(parts) - 1].split(';')[:(len(parts) - 1)])
            else:
                operand = parts[1]

            # annotations = ''.join(parts[len(parts) - 1].split(';')[-1].strip())
            pattern = r'\[(.*?)\]'
            match = re.search(pattern, operand)
            if match:
                content_within_brackets = match.group(1)
                common = [item for item in regs if item in content_within_brackets]
                operand = common[0]
                # print("Content within brackets:", common[0], content_within_brackets)
            triggers = []
            code = []
            param = {}
            if operand in regs:
                code.append(row)
                triggers.append(operand)
                for par in reversed(instr_set[:idx - 1]):
                    if 'call' in par and 'sub_' in par:
                        code.append(par)
                    if check_search_status(triggers, par):
                        vars = get_operands(triggers, par)
                        if vars:
                            for var in vars:
                                p_type = identify_parameter_type(var, regs)
                                if p_type == 3:
                                    if par not in code:
                                        code.append(par)
                                    if var not in triggers:
                                        triggers.append(var)
                                if p_type == 4:
                                    reg = extract_reg(var, regs)
                                    if reg:
                                        if par not in code:
                                            code.append(par)
                                        if reg not in triggers:
                                            triggers.append(reg)

                # param['p_annot'] = annotations
                param['p_val'] = operand
                param['tracked_code'] = code
                # print(*code,sep = "\n")
                # print('-------------------------------------')
            else:
                # param['p_annot'] = annotations
                param['p_val'] = operand
                param['tracked_code'] = None
            p_fingerprints.append(param)
    api_det["params"] = p_fingerprints
    return api_det


def extract_parameters(count, sample, fname, api_name, instr_list):
    params = []
    fn = None
    lib_call = False
    for instr in reversed(instr_list):
        ins = re.sub('\s+', ', ', instr, 1)
        parts = ins.split(', ')
        mnem = parts[0]

        if mnem == 'push' and ';' in parts[len(parts) - 1]:
            param = parts[len(parts) - 1].split(';')
            param_ = param[len(param) - 1].strip()
            if param[0].isnumeric():
                params.append(param_ + ' : ' + param[0])
            else:
                params.append(param_)
    if params:
        fn = {'no': count, 'hash': sample, 'fname': fname, 'api_name': api_name, 'params': params}
        append_record(fn)
        lib_call = True

    return lib_call, fn


def preprocess_process(count, sam, fname, df, data):
    instr_set = []
    for index, row in df.iterrows():
        row = list(row)
        if row[2] is not None:
            if 'call' == row[2].split(' ')[0]:  ##if 'call' in row[2]:
                instr_set.append(row[2])
                try:
                    api_name = row[2].split(" ")[4]
                except IndexError:
                    continue
                if 'sub_' in api_name:  ## extract contextual information
                    # instr_set.append(row[2])
                    continue

                api_dets = track_parameters(sam, fname, api_name, instr_set)
                # append_record(api_dets)
                print(api_dets)
                instr_set = []
            else:
                instr_set.append(row[2])


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# pd.set_option('display.max_colwidth', -1)

def preprocess_for_stripped_binaries(count, sam, fname, df, data, fill_mask):
    codeprint = []
    instr_set = []
    api_cp = []
    for index, row in df.iterrows():
        row = list(row)
        if row[2] is not None:
            if 'call' == row[2].split(' ')[0]:  ##if 'call' in row[2]:
                instr_set.append(row[2])
                try:
                    api_name_raw = row[2].split(" ")[4]
                    english_letters = re.findall('[a-zA-Z]+', api_name_raw)
                    api_name = ''.join(english_letters)
                except IndexError:
                    continue
                if 'sub_' in api_name:  ## extract contextual information
                    # instr_set.append(row[2])
                    continue
                # print(instr_set)
                api_fp = track_parameters_for_stripped_binaries(sam, fname, api_name, instr_set)
                api_cp = api_codeprint(api_fp)
                codeprint = api_cp
                if '[' in api_name_raw and ']' in api_name_raw:
                    print(api_name_raw, re.findall('[a-zA-Z]+', api_name_raw))
                    actual, predicted, corr, context_res, valid_input = predict_api(codeprint, fill_mask)
                    print(codeprint)
                    print("Actual: ", actual + "\n", "Predicted: " + predicted + "\n")
                append_record1(codeprint)
                instr_set = []
            else:
                instr_set.append(row[2])


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

def api_codeprint(api):
    api_n = api["api_name"]
    api_fp = []
    # if 'dword' not in api_n and '?' not in api_n and '_' not in api_n and '@' not in api_n and 'eax' not in api_n and 'ecx' not in api_n \
    #         and 'edi' not in api_n and 'edx' not in api_n and 'ebp' not in api_n and 'esp' not in api_n and 'ebx' not in api_n and 'esi' not in api_n:
    #     if 'ds:' in api_n:
    #         api_n = api_n.split(':')[1:]
    #         api_n = ' '.join(api_n)

    api_fp.append(api_n)
    for p in api['params']:
        if p['tracked_code']:
            instrs = []
            for instr in p['tracked_code']:
                str = re.sub(' +', ' ', instr)
                if 'sub_' in str:
                    f_call = str[:len('sub_')] + ' extrfun'
                    instrs.append(f_call)
                elif 'dword_' in str:
                    l = str.split(' ')
                    str = [str.replace(s, 'ptr') for i, s in enumerate(l) if 'dword_' in s.strip()]
                    instrs.append(' '.join(str))
                else:
                    cr_inst = []
                    inst = str.split(';')[0]
                    inst_ = inst.split(' ')
                    cr_inst.append(inst_[0])
                    for par in inst_[1:]:
                        val = par
                        p_type = identify_parameter_type(par, regs)
                        val = getnormalizedvalue(p_type, val)
                        cr_inst.append(val)
                    instrs.append(' '.join(cr_inst))
            # rec = [p["p_annot"], ' '.join(instrs)]
            rec = [' '.join(instrs)]
            api_fp.append(' '.join(rec))
        else:
            par = p["p_val"]
            p_type = identify_parameter_type(par, regs)
            val = normalizefixedparam(p_type, par, p)
            # rec = [p["p_annot"], val]
            rec = [val]
            api_fp.append(' '.join(rec))
    # append_record1(api_fp)
    return api_fp


def ishex(s):
    return not re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", s)


def identify_parameter_type(param, regs):
    p_type = 0
    regex = r"\[(.*?)\]"
    if param.isnumeric():
        p_type = 1

    matches = re.finditer(regex, param, re.MULTILINE)
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        txt = match.group()
        # print("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum=matchNum, start=match.start(), end=match.end(), match=match.group()))
        p_type = 4

    if param in regs:
        p_type = 3

    return p_type


def getnormalizedvalue(p_type, par):
    val = par
    if p_type == 0 and len(par.split(' ')) == 1:
        try:
            b = int(par.split('h')[0], 16)
            sz = int(math.log10(b) + 1)
            if sz <= 2:
                val = "saddr"
            elif sz > 2 and sz <= 4:
                val = "maddr"
            else:
                val = "laddr"
        except ValueError:
            if 'dword_' in par:
                l = par.split(' ')
                val = ' '.join([par.replace(s, 'ptr') for i, s in enumerate(l) if
                                'dword_' in s.strip()])
    if p_type == 4:
        val = par
        if len(val.split('+')) <= 2:
            val = "mem"
        else:
            val = "complex"
    if p_type == 1:
        val = par

    if p_type == 0 and 'unk_' in par:
        val = "unknown ptr"

    if p_type == 0 and 'offset' in par:
        val = "ptr"

    if p_type == 0 and 'off_' in par:
        val = "runtime ptr"

    return val


def normalizefixedparam(p_type, par, p):
    val = par
    if p_type == 0 and len(par.split(' ')) == 1:
        try:
            b = int(par.split('h')[0], 16)
            sz = int(math.log10(b) + 1)
            if sz <= 2:
                val = "saddr"
            elif sz > 2 and sz <= 4:
                val = "maddr"
            else:
                val = "laddr"
        except ValueError:
            if 'dword_' in par:
                l = par.split(' ')
                val = ' '.join(
                    [par.replace(s, 'ptr') for i, s in enumerate(l) if 'dword_' in s.strip()])
    if p_type == 4:
        val = par
        if len(val.split('+')) <= 2:
            val = "mem"
        else:
            val = "complex"
    if p_type == 1:
        val = p["p_val"]

    if p_type == 0 and 'unk_' in par:
        val = "unknown ptr"

    if p_type == 0 and 'offset' in par:
        val = "ptr"

    if p_type == 0 and 'off_' in par:
        val = "runtime ptr"

    return val


def cleandata(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = text.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words = re.sub(' +', ' ', ' '.join(words))
    return words


def append_record1(record):
    with open('api-codeprints-for-llm/test_for_rinser_llm.txt', 'a', newline="", encoding="utf-8") as f:
        f.write(cleandata(' '.join(record).rstrip()) + os.linesep)


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
        break
    return actual, predicted, corr, contxt_res, valid_input


def get_cos_sim(nlp_features, gt_api, predicted_api):
    output = nlp_features(gt_api)
    d = np.array(output)  # (Samples, Tokens, Vector Size)
    gt_api_bert = d[0].mean(axis=0)

    output = nlp_features(predicted_api)
    d = np.array(output)  # (Samples, Tokens, Vector Size)
    pred_api_bert = d[0].mean(axis=0)
    cos_sim = dot(pred_api_bert, gt_api_bert) / (norm(pred_api_bert) * norm(gt_api_bert))

    return cos_sim, gt_api_bert, pred_api_bert


def load_llm():
    tokenizer = BertTokenizerFast.from_pretrained(r'rinser-llm',
                                                  config=AutoConfig.from_pretrained(r'rinser-llm/checkpoint-28700'))
    model = BertForMaskedLM.from_pretrained(r'rinser-llm/checkpoint-28700/')

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    nlp_features = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    return tokenizer, model, fill_mask, nlp_features


def load_finetuned_llm():
    tokenizer = BertTokenizerFast.from_pretrained(r'rinser-llm',
                                                  config=AutoConfig.from_pretrained(r'rinser-llm/checkpoint-28700'))

    model = BertForMaskedLM.from_pretrained(r'rinser-llm\fine-tuned_for_non_annot_full_ds\checkpoint-25000')

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    nlp_features = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    return tokenizer, model, fill_mask, nlp_features


def get_codeprint(uni_params, api):
    api_fp = 0
    numparemeters = 0
    api_n = api["api_name"]
    if 'dword' in api_n or 'sub_' in api_n or api_n in regs:  # and '?' not in api_n and '_' not in api_n and '@' not in api_n and 'eax' not in api_n and 'ecx' not in api_n and 'edi' not in api_n and 'edx' not in api_n and 'ebp' not in api_n and 'esp' not in api_n and 'ebx' not in api_n and 'esi' not in api_n:
        if 'ds:' in api_n:
            api_n = api_n.split(':')[1:]
            api_n = ' '.join(api_n)

        # api_fp.append(api_n)
        if len(api['params']) > 0:
            api_fp = []
            for p in api['params']:
                if p['p_annot'] in uni_params:
                    numparemeters += 1
                    if p['tracked_code']:
                        instrs = []
                        for instr in p['tracked_code']:
                            sstr = re.sub(' +', ' ', instr)
                            if 'sub_' in sstr:
                                f_call = sstr[:len('sub_')] + ' extrfun'
                                instrs.append(f_call)
                            elif 'dword_' in sstr:
                                l = sstr.split(' ')
                                sstr = [sstr.replace(s, 'ptr') for i, s in enumerate(l) if 'dword_' in s.strip()]
                                instrs.append(' '.join(sstr))
                            else:
                                cr_inst = []
                                inst = sstr.split(';')[0]
                                inst_ = inst.split(' ')
                                cr_inst.append(inst_[0])
                                for par in inst_[1:]:
                                    val = par
                                    p_type = identify_parameter_type(par, regs)
                                    val = getnormalizedvalue(p_type, val)
                                    cr_inst.append(val)
                                instrs.append(' '.join(cr_inst))
                        rec = [p["p_annot"]]  # , ' '.join(instrs)]
                        api_fp.append(' '.join(rec))
                    else:
                        par = p["p_val"]
                        p_type = identify_parameter_type(par, regs)
                        val = normalizefixedparam(p_type, par, p)
                        rec = [p["p_annot"], val]
                        api_fp.append(' '.join(rec))
            if numparemeters > 0:
                api_fp.insert(0, str(numparemeters))  # only for testset with annot and code
                # append_record(api_fp, api["sample"])
                # print(api_n, api_fp, api["sample"])
    return api_fp


def get_gt_api(api_db, api_fp):
    gt_api = None
    tmp = None
    extracted_parameters_list = []
    codeprint = api_fp[1:]
    for p in codeprint:
        extracted_parameters_list.append(p.split(" ")[0])
    for k, v in api_db.items():
        # c_list = list(set(list(data[ind].split(' '))) & set(v["params"]))
        c_list = list(set(extracted_parameters_list) & set(v["params"]))
        if c_list:
            l_clist = len(c_list)
            if tmp is None:
                tmp = l_clist
                gt_api = k
                continue
            if tmp < l_clist:
                tmp = l_clist
                gt_api = k
    return gt_api


def model_prediction(out, api_db):
    predicted_api = None
    for prediction in out:
        if prediction["token_str"] not in api_db:
            continue
        predicted_api = prediction["token_str"]
        break
    return predicted_api
