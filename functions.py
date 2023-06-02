import re
import pandas as pd
from tqdm import tqdm
import os
import json
import warnings
import json
import os
import re
import string
import nltk
import math
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
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
                append_record(api_dets)
                instr_set = []
            else:
                instr_set.append(row[2])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)




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
