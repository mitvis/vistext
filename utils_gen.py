from zipfile import ZipFile
import os
from PIL import Image
import json
import re
import random
random.seed(2022)

def cleanL2L3(caption):
    caption = caption.strip()
    if caption[-1] != ".":
        caption = caption+"."
    return caption

def _wordsub(word_list):
    return [random.choice(word_list)]

def wordsub(in_sent, word_list):
    if len(word_list) > 1:
        return in_sent + _wordsub(word_list)
    else:
        word = word_list[0]
        if word == "This":
            choice = _wordsub(["This", "Here a"])
        elif word == "chart":
            choice = _wordsub(["chart", "graph", "diagram", "plot"])
        elif word == "titled":
            choice = _wordsub(["titled", "called", "named", "labeled"])
        elif word == "On":
            choice = _wordsub(["On", "Along"])
        elif word == "on":
            choice = _wordsub(["on", "along"])
        elif word == "plotted":
            choice = _wordsub(["plotted", "defined", "measured", "drawn", "shown"])
        elif word == "plots":
            choice = _wordsub(["plots", "measures", "shows"])
        elif word == "with":
            choice = _wordsub(["with", "using", "on", "along", "as"])
        elif word == "found":
            choice = _wordsub(["found", "seen"])
        elif word == "labeled":
            choice = _wordsub(["labeled", "marked"])
        else:
            choice = [word]
        return in_sent + choice
    
def generate_title(d_charttype, d_title):
    basic_titles = {1: f"This is a {d_charttype} titled {d_title}",
                    2: f"This {d_charttype} is titled {d_title}",
                    3: f"{d_title} is a {d_charttype}"
                   }
    select = random.choice(list(basic_titles.items()))
    
    sentence = []
    if select[0] == 1:
        #sentence = ["This", "is", "a", d_charttype]
        sentence = wordsub(sentence, ["This"])
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["a"])
        sentence.append(d_charttype)
        sentence = wordsub(sentence, ["chart"])
        sentence = wordsub(sentence, ["titled"])
        sentence.append(d_title)
        
    elif select[0] == 2:
        sentence = wordsub(sentence, ["This"])
        sentence.append(d_charttype)
        sentence = wordsub(sentence, ["chart"])
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["titled"])
        sentence.append(d_title)
        
    elif select[0] == 3:
        sentence.append(d_title)
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["a"])
        sentence.append(d_charttype)
        sentence = wordsub(sentence, ["chart"])
    
    return " ".join(sentence) + '.'

def generate_axis(d_axis, d_axis_label, d_axis_scale):
    basic_axis = {1: f"On the {d_axis}, {d_axis_label} is plotted with a {d_axis_scale}",
                  2: f"{d_axis_label} is plotted with a {d_axis_scale} on the {d_axis}",
                  3: f"The {d_axis} plots {d_axis_label} with a {d_axis_scale}",
                  4: f"A {d_axis_scale} can be found on the {d_axis}, labeled {d_axis_label}",
                  5: f"This is a {d_axis_scale} on the {d_axis}, labeled {d_axis_label}",
                 }
    select = random.choice(list(basic_axis.items()))
    
    sentence = []
    if select[0] == 1:
        sentence = wordsub(sentence, ["On"])
        sentence = wordsub(sentence, ["the"])
        sentence.append(d_axis+',')
        sentence.append(d_axis_label)
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["plotted"])
        if bool(random.getrandbits(1)) == True:
            sentence = wordsub(sentence, ["with"])
            sentence = wordsub(sentence, ["a"])
            sentence.append(d_axis_scale)

    if select[0] == 2:
        sentence.append(d_axis_label)
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["plotted"])
        if bool(random.getrandbits(1)) == True:
            sentence = wordsub(sentence, ["with"])
            sentence = wordsub(sentence, ["a"])
            sentence.append(d_axis_scale)
        sentence = wordsub(sentence, ["on"])
        sentence = wordsub(sentence, ["the"]) 
        sentence.append(d_axis)
    
    if select[0] == 3:
        sentence = wordsub(sentence, ["The"])
        sentence.append(d_axis)
        sentence = wordsub(sentence, ["plots"])
        sentence.append(d_axis_label)
        if bool(random.getrandbits(1)) == True:
            sentence = wordsub(sentence, ["with"])
            sentence = wordsub(sentence, ["a"])
            sentence.append(d_axis_scale)
            
    if select[0] == 4:
        sentence = wordsub(sentence, ["A"])
        sentence.append(d_axis_scale)
        sentence = wordsub(sentence, ["can"])
        sentence = wordsub(sentence, ["be"])
        sentence = wordsub(sentence, ["found"])
        sentence = wordsub(sentence, ["on"])
        sentence = wordsub(sentence, ["the"])
        sentence.append(d_axis+',')
        sentence = wordsub(sentence, ["labeled"])
        sentence.append(d_axis_label)
        
    if select[0] == 5:
        sentence = wordsub(sentence, ["There"])
        sentence = wordsub(sentence, ["is"])
        sentence = wordsub(sentence, ["a"])
        sentence.append(d_axis_scale)
        sentence = wordsub(sentence, ["on"])
        sentence = wordsub(sentence, ["the"])
        sentence.append(d_axis+',')
        sentence = wordsub(sentence, ["labeled"])
        sentence.append(d_axis_label)
            
    return " ".join(sentence) + '.'

def generate_jointaxis(d_axis_x, d_axis_y, d_axis_label_x, d_axis_label_y, d_axis_scale_x, d_axis_scale_y):
    if bool(random.getrandbits(1)) == True:
        d_axis_1, d_axis_label_1, d_axis_scale_1 = d_axis_x, d_axis_label_x, d_axis_scale_x
        d_axis_2, d_axis_label_2, d_axis_scale_2 = d_axis_y, d_axis_label_y, d_axis_scale_y
    else:
        d_axis_1, d_axis_label_1, d_axis_scale_1 = d_axis_y, d_axis_label_y, d_axis_scale_y
        d_axis_2, d_axis_label_2, d_axis_scale_2 = d_axis_x, d_axis_label_x, d_axis_scale_x

    basic_jointaxis = {1: f"The {d_axis_1} plots {d_axis_label_1} with {d_axis_scale_1} while the {d_axis_2} plots {d_axis_label_2} with {d_axis_scale_2}"}
    
    select = random.choice(list(basic_jointaxis.items()))

    sentence = []
    if select[0] == 1:
        usescale = bool(random.getrandbits(1))
        
        sentence = wordsub(sentence, ["The"])
        sentence.append(d_axis_1)
        sentence = wordsub(sentence, ["plots"])
        sentence.append(d_axis_label_1)
        if usescale:
            sentence = wordsub(sentence, ["with"])
            sentence.append(d_axis_scale_1)
        sentence = wordsub(sentence, ["while"])
        sentence = wordsub(sentence, ["the"])
        sentence.append(d_axis_2)
        sentence = wordsub(sentence, ["plots"])
        sentence.append(d_axis_label_2)
        if usescale:
            sentence = wordsub(sentence, ["with"])
            sentence.append(d_axis_scale_2)
            
    return " ".join(sentence) + '.'

def getmetadataloc(findme, longstring):
    startloc = longstring.find(findme)
    if startloc != -1:
        return (startloc, startloc+len(findme))
    else:
        return (None, None)

def generate_caption(d_charttype, d_title, d_axis_label_x, d_axis_label_y, d_axis_scale_x, d_axis_scale_y):
    usejointaxis = bool(random.getrandbits(1))
    
    d_axis_x = "x-axis"
    d_axis_y = "y-axis"
    
    cap_title = generate_title(d_charttype, d_title)
    if usejointaxis:
        cap_axis = generate_jointaxis(d_axis_x, d_axis_y, d_axis_label_x, d_axis_label_y, d_axis_scale_x, d_axis_scale_y)
    else:
        cap_axis_x = generate_axis(d_axis_x, d_axis_label_x, d_axis_scale_x)
        cap_axis_y = generate_axis(d_axis_y, d_axis_label_y, d_axis_scale_y)
        if bool(random.getrandbits(1)):
            cap_axis = " ".join([cap_axis_x, cap_axis_y])
        else:
            cap_axis = " ".join([cap_axis_y, cap_axis_x])
    
    caption = [cap_title, cap_axis]
#     random.shuffle(caption)
    caption = " ".join(caption)
    
    metadata = {}
    metadata['charttype'] = d_charttype
    metadata['loc_charttype'] = getmetadataloc(d_charttype, caption)
    metadata['loc_title'] = getmetadataloc(d_title, caption)
    metadata['loc_axis_label_x'] = getmetadataloc(d_axis_label_x, caption)
    metadata['loc_axis_label_y'] = getmetadataloc(d_axis_label_y, caption)
    metadata['loc_axis_scale_x'] = getmetadataloc(d_axis_scale_x, caption)
    metadata['loc_axis_scale_y'] = getmetadataloc(d_axis_scale_y, caption)
    
    return (caption, metadata)

def json_extract_key(obj, key, subtree=False):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append((v, dict(obj.items())))
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    if subtree==False:
        return [x[0] for x in values]
    else:
        return values
    
def json_extract_value(obj, key, subtree=False):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif v == key:
                    arr.append((v, dict(obj.items())))
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    if subtree==False:
        return [x[0] for x in values]
    else:
        return values
    
def json_extract_key_multiple(obj, key, subtree=False):
    """Recursively fetch values from nested JSON."""
    arr = {}

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k in key:
                    arr.update({k: v})
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    
    return values

def flatten_dict(i_dict):
    flattened = []
    for k, v in i_dict.items():
        flattened.append(str(k))
        if type(v) == float:
            flattened.append(str(round(v,3)))
        else:
            flattened.append(str(v))
#         if len(str.split(str(v))) == 1:
#             flattened.append(str(v))
#         else:
# #             flattened.append("'"+str(v)+"'")
#             flattened.append(str(v))
    return flattened

def parse_title(obj, keys=["x", "y", "text"], src=True):
    parsed = []
    title = json_extract_value(obj, "title", subtree=True)[0][1]

    parsed.append(json_extract_key_multiple(title, keys))

#     print(parsed)
    parsed = flatten_dict(parsed[0])
    keys_trunc = [k for k in keys if k not in ["x", "y"]]

    titletext =  " ".join(json_extract_value(title, "text", subtree=True)[0][1]["items"][0]["text"])
#     print(titletext)

    parsed = ["title"] + [titletext] + [x for x in parsed if x not in keys_trunc]
    
    if src:
        result = parsed
    
    else:
        result = titletext
    
    return result

def parse_axes(obj, keys=["x", "y", "text"], src=True):
    parsed_list = []
    axes = json_extract_value(obj, "axis-title", subtree=True)
    for i, ax in enumerate(axes):
        extracted = json_extract_key_multiple(ax[1], keys)
        if i == 0:
            biggest_x = extracted['x']
            parsed_list.append(extracted)
        elif i == 1:
            if extracted['x'] > biggest_x:
                parsed_list = [extracted] + parsed_list
            else:
                parsed_list.append(extracted)
        else:
            print("err")
    
    if src:
        keys_trunc = [k for k in keys if k not in ["x", "y"]]
        parsed = []

    #     for i in parsed_list:
    #         parsed = parsed + [x for x in flatten_dict(i) if x not in keys_trunc]

        parsed = ["x-axis"] + [x for x in flatten_dict(parsed_list[0]) if x not in keys_trunc] + ["y-axis"] + [x for x in flatten_dict(parsed_list[1]) if x not in keys_trunc]
    
    else:
        parsed_x = json_extract_key(parsed_list[0], "text", subtree=False)[0]
        parsed_y = json_extract_key(parsed_list[1], "text", subtree=False)[0]
        parsed = {'x-axis': parsed_x, 'y-axis': parsed_y}

    return parsed

def parse_ticks(obj, keys=["x", "y"]):
    parsed_list_ticks = []
    axes_ticks = json_extract_value(obj, "axis-tick", subtree=True)
#     if len(axes_ticks) == 1:
#         alt_ticks = True
#     else:
#         alt_ticks = False
    parsed_list_labels = []
    axes_labels = json_extract_value(obj, "axis-label", subtree=True)

    for i, ax in enumerate(axes_ticks):
#         extracted = json_extract_key_multiple(ax[1], keys)
        extracted = [json_extract_key_multiple(x, keys) for x in ax[1]['items']]

        if extracted[0]['y'] == extracted[-1]['y']:
            parsed_list_ticks.insert(0, extracted)
        else:
            parsed_list_ticks.insert(1, extracted)

    for i, ax in enumerate(axes_labels):
        extracted = [json_extract_key_multiple(x, keys+["text"]) for x in ax[1]['items']]

        if extracted[0]['y'] == extracted[-1]['y']:
            parsed_list_labels.insert(0, extracted)
        else:
            parsed_list_labels.insert(1, extracted)

    if len(parsed_list_ticks) < 2:
        ys = [i['y'] for i in parsed_list_labels[1]]
        ys.reverse()
        y_reorder = []
        for idx, dat in enumerate(parsed_list_labels[1]):
            dat_reorder = dat
            dat_reorder['y'] = ys[idx]
            y_reorder.append(dat_reorder)
        
        zipped_x = zip(parsed_list_labels[0], parsed_list_labels[0])
        zipped_y = zip(parsed_list_labels[1], parsed_list_labels[1])
    else:
        zipped_x = zip(parsed_list_ticks[0], parsed_list_labels[0])
        zipped_y = zip(parsed_list_ticks[1], parsed_list_labels[1])
    
#     xticks = [["x", str(x[0]['x']), "val", "'"+x[1]['text']+"'"] for x in zipped_x]
#     yticks = [["y", str(y[0]['y']), "val", "'"+y[1]['text']+"'"] for y in zipped_y]
    xticks = [["x", str(x[0]['x']), "val", x[1]['text']] for x in zipped_x]
    yticks = [["y", str(y[0]['y']), "val", y[1]['text']] for y in zipped_y]
        
#     parsed = []
#     xticks = [flatten_dict(x) for x in parsed_list[0]]
    xticks = [item for sublist in xticks for item in sublist]
#     yticks = [flatten_dict(x) for x in parsed_list[1]]
    yticks = [item for sublist in yticks for item in sublist]
    parsed = ["xtick"] + xticks + ["ytick"] + yticks

    return parsed

def flatten_dict_marks(i_dict):
    flattened = []
    for k, v in i_dict.items():
#         print(f"processing {k} {v}")
        if k == 'x':
            flattened.append("XY")
#             print(f"appending xy")
        elif k == 'y':
            pass
#             print(f"passing on y")
        else:
            k = str(k)
            if k == "height":
                k = "H"
            elif k == "description":
                k = "desc"
            flattened.append(k)
#             print(f"appending {k}")
            
        if len(str.split(str(v))) == 1:
            if type(v) == float:
                flattened.append(str(round(v,3)))
            else:
                flattened.append(str(v))
    return flattened

def parse_marks(obj, keys=["x", "y", "width", "height", "description"], src=True):
    parsed_list = []
    marks = json_extract_value(obj, "mark", subtree=True)[0][1]
#     parsed_list.append(json_extract_key_multiple(marks, ["marktype"]))

    if src:
        for mark in marks['items']:
            parsed_list.append(json_extract_key_multiple(mark, keys))

        keys_trunc = [k for k in keys if k not in ["x", "y", "width", "height"]]
        parsed = []
        parsed.append("marks")
        charttype = json_extract_key(marks, "marktype")[0]
        if charttype == 'symbol':
            charttype = 'scatter'
        elif charttype == 'rect':
            charttype = 'bar'
        parsed.append(charttype)

#         print(parsed_list[0])
        for i in parsed_list:
            parsed = parsed + [x for x in flatten_dict_marks(i) if x not in keys_trunc]
    
    else:
        # area, line, symbol, rect, 
        parsed = json_extract_key(marks, "marktype")[0]
        if parsed == 'symbol':
            parsed = 'scatter'
        elif parsed == 'rect':
            parsed = 'bar'
    
    return parsed

def parse_all_sg(obj, flatten=True, src=True, prettyprint=False):
    if src:
        parse = parse_title(obj) + parse_axes(obj) + parse_ticks(obj) + parse_marks(obj)
    #     parse_flattened = []
    #     for i in parse_joint:
    #         parse_flattened = parse_flattened + flatten_dict(i)
        if flatten:
            parse = " ".join(parse)
        
        if prettyprint:
            print(" ".join(parse_title(obj)))
            print(" ".join(parse_axes(obj)))
            print(" ".join(parse_ticks(obj)))
            print(" ".join(parse_marks(obj)))
        
    else:
#         generate_caption(d_charttype, d_title, d_axis_label_x, d_axis_label_y, d_axis_scale_x, d_axis_scale_y)
        parse = []
        parse.append(parse_marks(obj, src=False))
        parse.append(parse_title(obj, src=False))
        parse += [parse_axes(obj, src=False)['x-axis'], parse_axes(obj, src=False)['y-axis']]
        parse += [parse_scales(obj)['x-scale'], parse_scales(obj)['y-scale']]
    return parse

def parse_dataline_dt(datum, xname, yname):
#     datumline = datum.split(';')
    idx_xname = datum.find(xname)
    x_len = len(xname)
    idx_semi = datum.find(";", idx_xname+x_len)
    idx_yname = datum.find(yname, idx_semi)
    y_len = len(yname)
    datum_x = datum[idx_xname+x_len+2:idx_semi]
    datum_y = datum[idx_yname+y_len+2:]
    return datum_x, datum_y

def parse_marks_dt(obj, chart_x, chart_y):
    mark_items = json_extract_value(obj, "mark", subtree=True)[0][1]['items']
    chart_data = [x['description'] for x in mark_items]
    chart_data_parsed = [parse_dataline_dt(x, chart_x, chart_y) for x in chart_data]
    return chart_data_parsed

def parse_all_dt(obj):
    chart_title = parse_title(obj)[1]
    chart_x = parse_axes(obj)[5]
    chart_y = parse_axes(obj)[11]
    chart_data_parsed = parse_marks_dt(obj, chart_x, chart_y)
    chart_data_parsed = " ".join([x[0]+" "+x[1] for x in chart_data_parsed])
    parsed = chart_title + " <s> " + chart_x + " " + chart_y + " " + chart_data_parsed
    return parsed

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def parse_scales(obj):
    axes_scales = json_extract_value(obj, "axis-label", subtree=True)
    firstscale = axes_scales[0][1]['items']
    secondscale = axes_scales[1][1]['items']
    
    if (firstscale[0]['y'] == firstscale[-1]['y']) & (firstscale[0]['x'] != firstscale[-1]['x']):
        axes = [firstscale, secondscale]
    else:
        axes = [secondscale, firstscale]
    
    scale_strings = []
    for ax in axes:
        if is_number(ax[0]['text'].replace(',','')) and is_number(ax[1]['text'].replace(',','')) and is_number(ax[-1]['text'].replace(',','')):
            delta = float(ax[1]['text'].replace(',','')) - float(ax[0]['text'].replace(',',''))
            delta = round(delta) if delta.is_integer() else delta
            if (len(ax)-1)*delta == (float(ax[-1]['text'].replace(',','')) - float(ax[0]['text'].replace(',',''))):
                poss_scales = ["linear scale of range "+ax[0]['text']+" to "+ax[-1]['text'],
                               #"linear scale from range "+ax[0]['text']+" to "+ax[-1]['text']+" with an interval of "+str(delta),
                               "linear scale with a minimum of "+ax[0]['text']+" and a maximum of "+ax[-1]['text'],
                               "linear scale from "+ax[0]['text']+" to "+ax[-1]['text']
                              ]
                scale_strings.append(random.choice(poss_scales))
            else:
                poss_scales = ["scale of range "+ax[0]['text']+" to "+ax[-1]['text'],
                               "scale with a minimum of "+ax[0]['text']+" and a maximum of "+ax[-1]['text'],
                               "scale from "+ax[0]['text']+" to "+ax[-1]['text']
                              ]
                scale_strings.append(random.choice(poss_scales))
        else:
            poss_scales = ["categorical scale from "+ax[0]['text']+" to "+ax[-1]['text'],
                           "categorical scale starting at "+ax[0]['text']+" and ending at "+ax[-1]['text'],
                           "categorical scale starting with "+ax[0]['text']+" and ending with "+ax[-1]['text'],
                           "categorical scale with "+ax[0]['text']+" on one end and "+ax[-1]['text'] + " at the other",
                          ]
            scale_strings.append(random.choice(poss_scales))
            
        # how to do temporal scale?
        
#     x_axis_scale = "scale from range " + str(axes[0][0]['text']) + " to " + str(axes[0][-1]['text'])
#     y_axis_scale = "scale from range " + str(axes[1][0]['text']) + " to " + str(axes[1][-1]['text'])
    
    return {'x-scale': scale_strings[0], 'y-scale': scale_strings[1]}