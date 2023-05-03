def tokenizerEncDec(iput, tokenizer):
    enc = tokenizer.encode(iput)
    dec = tokenizer.decode(enc.ids)
    return dec
    
def evalRG_single(prediction, data):
    pr_TP = sum([1 for item in data if item in prediction])
    fields = (1 if "y - axis" in prediction else 0) + (1 if "x - axis" in prediction else 0) + 2
    pr_FN = fields - pr_TP
    
    pr_TN = 0
    pr_FP = 0
    
    precision = pr_TP/(pr_TP+pr_FN)
    recall = 0
    
    return precision

def evalRG(predictions, datas):
    RG_all = []
    for idx, datum in enumerate(datas):
        RG_all.append(evalRG_single(predictions[idx], datum))
    return RG_all

def wmd_preprocess(sentence, stop_words):
    return [w for w in sentence.lower().split() if w not in stop_words]

def evalWMD_single(prediction, data, w2v_model, stop_words):
    distance = w2v_model.wmdistance(wmd_preprocess(prediction, stop_words), wmd_preprocess(data, stop_words))
    return distance

def evalWMD(predictions, datas, w2v_model, stop_words):
    WMD_all = []
    for idx, datum in enumerate(datas):
        WMD_all.append(evalWMD_single(predictions[idx], datum, w2v_model, stop_words))
    return WMD_all

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]