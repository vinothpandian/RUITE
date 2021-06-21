import torch
import pandas as pd
from torchtext import data
from torchtext.data import Field
from torchtext.data import Example
import torch.nn as nn
from models.model import Transformer
import matplotlib
import matplotlib.pyplot as plt


def refineUI(UIdict):
    elt_data = pd.DataFrame(columns=['uiid','layout_ht','layout_wd','elt','x1','y1','ht','wd'])
    num_elts = 0
    for layout in UIdict:
        uiid = layout['id']
        layout_ht = layout['height']
        layout_wd = layout['width']
        for elt in layout['objects']:
            num_elts=num_elts+1
            category = elt['name']
            x = elt['position']['x']
            y = elt['position']['y']
            width = elt['dimension']['width']
            height = elt['dimension']['height']
            elt_data = elt_data.append({'uiid':uiid, 'layout_ht':layout_ht, 'layout_wd':layout_wd, 'elt':category, 'x1':x, 'y1':y, 'ht':height, 'wd':width},ignore_index=True)

    elt_data = elt_data.sort_values(['uiid', 'y1', 'x1']).reset_index()
    elt_data = elt_data.drop(columns=['index'])

    elt_data['xmin_int']=((elt_data['x1']/elt_data['layout_wd'])*15).astype(int)
    elt_data['width'] = ((elt_data['wd'] / elt_data['layout_wd']) * 15).astype(int)
    elt_data['ymin_int'] = ((elt_data['y1'] / elt_data['layout_ht']) * 25).astype(int)
    elt_data['height'] = ((elt_data['ht'] / elt_data['layout_ht']) * 25).astype(int)

    elt_data['xmin_int'] = 'xm' + elt_data['xmin_int'].astype(str)
    elt_data['ymin_int'] = 'ym' + elt_data['ymin_int'].astype(str)
    elt_data['height'] = 'ht' + elt_data['height'].astype(str)
    elt_data['width'] = 'wd' + elt_data['width'].astype(str)

    elt_data['seq_x']= elt_data['elt'].astype(str) + ' '+elt_data['xmin_int'].astype(str) + ' ' + elt_data['ymin_int'].astype(str) + ' ' + \
                    elt_data['height'].astype(str) + ' ' + elt_data['width'].astype(str)

    elt_data = elt_data.groupby(['uiid'])['seq_x'].apply(' '.join).reset_index()

    SEQ_X_test = Field(sequential=True, use_vocab=True, fix_length=100)

    datafields = [("seq_x", SEQ_X_test)]

    data_set = Example.fromlist(
        data=elt_data['seq_x'],
        fields = datafields
    )
    print("input:",data_set.seq_x)
    import os
    #print(os.getcwd())
    vocab = torch.load('../results/vocab9.tar')
    #model = torch.load('../results/model2.pt',map_location=torch.device("cpu"))
    model = Transformer(vocab_size=113, d_model=32, N=2, heads=2)
    model.load_state_dict(torch.load('../results/model9.tar',map_location=torch.device("cpu")))
    model.eval()
    #print(vocab.itos[1])
    data_set.vocab = vocab
    transformed_input = [vocab.stoi[element] for element in data_set.seq_x]
    print("input_transformed:", transformed_input)
    if(len(transformed_input) < 100):
        padding_len = 100-len(transformed_input)
        transformed_input = transformed_input + [1]*padding_len
    else:
        transformed_input = transformed_input[:100]
    transformed_input = torch.LongTensor(transformed_input).unsqueeze(1)
    for_mask = transformed_input.transpose(0,1)
    pad = 1
    input_msk = (for_mask == pad)
    #print(input_msk)
    #print(transformed_input)
    print(transformed_input.size())
    output_logits = model(transformed_input, input_msk)
    print(output_logits.size())
    softmax = nn.Softmax(dim=-1)
    prediction = softmax(output_logits).argmax(2)
    print("output_transformed:",prediction[0])
    actual_preds = [vocab.itos[p] for p in prediction[0]]
    print("output:",actual_preds[0:(num_elts*6)])


    output_dict= UIdict.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #seq_x = ' '.join(UIdict)
    # fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
    ax1.axis('off')
    ax2.axis('off')
    for i in range(num_elts):

        rect1 = matplotlib.patches.Rectangle((output_dict[0]["objects"][i]["position"]["x"], output_dict[0]["objects"][i]["position"]["y"]),
                                             output_dict[0]["objects"][i]["dimension"]["width"], output_dict[0]["objects"][i]["dimension"]["height"],
                                             color='black', fill=0)
        ax1.add_patch(rect1)

        output_dict[0]["objects"][i]["name"] = actual_preds[i*6]
        if ((actual_preds[i*6+1][2:]).isnumeric()):
            xm_new = (int(actual_preds[i*6+1][2:])/15*layout_wd) + (layout_wd/20)
            output_dict[0]["objects"][i]["position"]["x"] = xm_new
        if ((actual_preds[i*6+2][2:]).isnumeric()):
            ym_new = (int(actual_preds[i*6+2][2:])/25*layout_ht) + (layout_ht/30)
            output_dict[0]["objects"][i]["position"]["y"] = ym_new
        if ((actual_preds[i * 6 + 3][2:]).isnumeric()):
            ht_new = (int(actual_preds[i*6+3][2:])/25*layout_ht) + (layout_ht/30)
            if (ym_new+ht_new >layout_ht):
                ht_new = layout_ht - ym_new
            output_dict[0]["objects"][i]["dimension"]["height"] = ht_new
        if ((actual_preds[i * 6 + 4][2:]).isnumeric()):
            wd_new = (int(actual_preds[i*6+4][2:])/15*layout_wd) + (layout_wd/20)
            if (wd_new+xm_new >layout_wd):
                wd_new = layout_wd - xm_new
            output_dict[0]["objects"][i]["dimension"]["width"] = wd_new
        if ((actual_preds[i * 6 + 5]).isnumeric()):
            group = int(actual_preds[i*6+5])
            output_dict[0]["objects"][i]["group"] = group


        rect2 = matplotlib.patches.Rectangle((xm_new, ym_new),
                                             wd_new, ht_new,
                                             color='red', fill=0)
        ax2.add_patch(rect2)

    ax1.set_xlim([0, layout_wd+5])
    ax1.set_ylim([0, layout_ht+5])
    ax2.set_xlim([0, layout_wd+5])
    ax2.set_ylim([0, layout_ht+5])

    plt.show()

    print(output_dict)
    return output_dict


def main():

    thisdict = [
  {
    "id": "14dbcf65-cdbb-41a4-a997-68a67fb6f8bd",
    "width": 428,
    "height": 926,
    "objects": [
      {
        "name": "text_field",
        "position": {
          "x": 44,
          "y": 388
        },
        "dimension": {
          "width": 289,
          "height": 58
        }
      },
      {
        "name": "text_field",
        "position": {
          "x": 33,
          "y": 463
        },
        "dimension": {
          "width": 273,
          "height": 55
        }
      },
      {
        "name": "button",
        "position": {
          "x": 81,
          "y": 583
        },
        "dimension": {
          "width": 267,
          "height": 55
        }
      },
      {
        "name": "image",
        "position": {
          "x": 21,
          "y": 10
        },
        "dimension": {
          "width": 387,
          "height": 367
        }
      }
    ]
  }
]


    return refineUI(thisdict)
if __name__ == "__main__":
    hi = main()