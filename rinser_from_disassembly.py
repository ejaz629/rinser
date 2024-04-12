from functions import *

directory = r"disassembly"
func_limit = 0
sample_limit = 3000
api_fp = []
funcs_apis = []
regs = ['eax', 'esp', 'ecx', 'ebp', 'ebx', 'edx', 'esi', 'edi']
count = 1
file_sizes = []

for filename in os.listdir(directory):
    sample = os.path.join(directory, filename)
    track = False  
    print(sample)
    try:
        data = pd.read_csv(sample, sep=r" : ", skiprows=2, encoding="cp1252", header=None, index_col=False)[
           :-2]  
    except:
        pass
    print([count, 'extracting features...'])
    data.drop(data.tail(2).index, inplace=True)  # drop last n rows
    funcslist = list(data[0].unique())
    funs = 0
    for f in funcslist:
        df = data[data[0] == f]
        preprocess_process(count, filename, f, df, data)
        funs += 1
        if funs == func_limit:
            break
    count += 1
    # if count == 2:
    #     break

print("JSON formated file for the disassembly generated!")

hits = 0
api_count = 0
numparemeters = 0
apilist = []
directory = r"api-codeprints"
for filename in os.listdir(directory):
    sample = os.path.join(directory, filename)
    if os.path.isfile(sample):
        print(sample)
        for line in open(sample, 'r'):
            api = json.loads(line)
            api_n = api["api_name"]
            if 'dword' not in api_n and '?' not in api_n and '_' not in api_n and '@' not in api_n and 'eax' not in api_n and 'ecx' not in api_n \
                    and 'edi' not in api_n and 'edx' not in api_n and 'ebp' not in api_n and 'esp' not in api_n and 'ebx' not in api_n and 'esi' not in api_n:
                api_count += 1
                if 'ds:' in api_n:
                    api_n = api_n.split(':')[1:]
                    api_n = ' '.join(api_n)
                if api_n not in apilist:
                    apilist.append(api_n)
                    hits += 1
                api_fp = []
                api_fp.append(api_n)
                for p in api['params']:
                    numparemeters += 1
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
                        rec = [p["p_annot"], ' '.join(instrs)]
                        api_fp.append(' '.join(rec))
                    else:
                        par = p["p_val"]
                        p_type = identify_parameter_type(par, regs)
                        val = normalizefixedparam(p_type, par, p)
                        rec = [p["p_annot"], val]
                        api_fp.append(' '.join(rec))
                append_record1(api_fp)
                #print([api_n, rec])
print(hits)
print(api_count)
print(numparemeters)
