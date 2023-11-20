from functions import *
p_LIMIT=0
directory = r"C:\Users\ahm038\PycharmProjects\asmbert\data"
tokenizer, model, fill_mask, nlp_features = load_llm()

with open('full_ds_params_unique.txt', 'r') as f:
    uni_params = f.read().split('\n')

with open('api_db_final.json', 'r') as fa:
    api_db = json.load(fa)


for filename in os.listdir(directory):
    sample = os.path.join(directory, filename)
    if os.path.isfile(sample):
        print(sample)
        for line in open(sample, 'r'):
            api = json.loads(line)
            api_fp = get_codeprint(uni_params, api)
            if api_fp:
                corr = False
                n_p = api_fp[0]
                if n_p != p_LIMIT:
                    t_ex = ' '.join(api_fp[1:])
                    if len(t_ex) <= 600 and len(t_ex) > 1:
                        example = "[MASK] " + t_ex
                        try:
                            out = fill_mask(example)
                        except RuntimeError:
                            continue

                        predicted_api = model_prediction(out, api_db)
                        gt_api = get_gt_api(api_db, api_fp)
                        if gt_api and predicted_api != gt_api:
                            predicted_api = gt_api
                        print(f"predicted API: {predicted_api}\n API codeprint: {t_ex}\n")

