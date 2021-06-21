import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from models.dataset import UIdataset
from tensorboard.plot_images import *
from Utils.utils import *


def get_orignal_data(dataset, inp):
    l = []
    for i in inp:
        l.append(dataset.get_vocab_item(i))
    return l

def plot_images(writer, i_list, p_list, t_list, dataset, orig_data):
    #Reading all elements and saving in dict
    pred_df=pd.DataFrame(columns=['filename','x1', 'y1','ht','wt'])
    elementdict={
    "alert":plt.imread('../data/images/alert.png'),
    "button":plt.imread('../data/images/button.png'),
    "card":plt.imread('../data/images/card.png'),
    "checkbox_checked":plt.imread('../data/images/checkbox_checked.png'),
    "checkbox_unchecked":plt.imread('../data/images/checkbox_unchecked.png'),
    "chip":plt.imread('../data/images/chip.png'),
    "data_table":plt.imread('../data/images/data_table.png'),
    "floating_action_button":plt.imread('../data/images/floating_action_button.png'),
    "grid_list":plt.imread('../data/images/grid_list.png'),
    "image":plt.imread('../data/images/image.png'),
    "label":plt.imread('../data/images/label.png'),
    "dropdown_menu":plt.imread('../data/images/menu.png'),
    "menu": plt.imread('../data/images/menu.png'),
    "radio_button_checked":plt.imread('../data/images/radio_button_checked.png'),
    "radio_button_unchecked":plt.imread('../data/images/radio_button_unchecked.png'),
    "slider":plt.imread('../data/images/slider.png'),
    "switch_disabled":plt.imread('../data/images/switch_disabled.png'),
    "switch_enabled":plt.imread('../data/images/switch_enabled.png'),
    "text_area":plt.imread('../data/images/text_area.png'),
    "text_field":plt.imread('../data/images/text_field.png'),
    "tooltip":plt.imread('../data/images/tooltip.png')
    }

    colors = {
        "1": "blue",
        "2": "red",
        "3": "green",
        "4": "yellow",
        "5": "grey",
        "6": "black",
        "7": "cyan",
        "8": "brown"
    }
    # gt_data = {
    #     "info": {
    #         "year": 2021,
    #         "version": "1.0.0",
    #         "description": "RUITE: Refining UI layout Aesthetics using transformer encoder",
    #         "contributor": "Soliha Rahman",
    #         "url": "",
    #         "date_created": "2021/03/07"
    #     },
    #     "licenses": [
    #         {
    #             "id": 0,
    #             "name": "CC-BY-SA 4.0",
    #             "url": "https://creativecommons.org/licenses/by-sa/4.0/"
    #         }
    #     ],
    #     "annotations":[],
    #     "categories":[]}
    # print(gt_data)
    #gt_data['annotations']=[]
    #gt_data['categories']=[]
    #op_data=[]
    #ip_data=[]
    #ann_id=0
    for i in range(len(i_list)):
        inp = i_list[i]
        p = p_list[i]
        t = t_list[i]
        if((1 in inp) & UIconfig.group):
            index_1 = inp.index(1)
            inp = inp[:index_1]
            p = p[:int(index_1+((index_1+1)/5))]
            t = t[:index_1]
        elif(1 in inp):
            index_1 = inp.index(1)
            inp = inp[:index_1]
            p = p[:index_1]
            t = t[:index_1]
        inp_dict=get_orignal_data(dataset, inp)
        pred_dict=get_orignal_data(dataset, p)
        true_dict=get_orignal_data(dataset, t)
        print(true_dict)
        print(pred_dict)
        k=' '.join(inp_dict)
        #print(k)
        #print(orig_data[orig_data['seq_x']==k]['UIID'].values[0])

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # seq_x = ' '.join(inp_dict)
        # #fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
        # ax1.axes.xaxis.set_visible(False)
        # ax2.axes.xaxis.set_visible(False)
        # ax3.axes.xaxis.set_visible(False)
        #
        # ax1.axes.yaxis.set_visible(False)
        # ax2.axes.yaxis.set_visible(False)
        # ax3.axes.yaxis.set_visible(False)

        filename=str(i)+".txt"

        if ('<unk>' not in pred_dict) & ('<unk>' not in true_dict) & ('<unk>' not in inp_dict):
            import os

            #print(os.getcwd())
            gt_file = open("../results/GroundTruth/" + filename, "w")
            op_file = open("../results/Output/" + filename, "w")
            ip_file = open("../results/Input/" + filename, "w")
            for j in range(0, int(np.floor(len(inp) / 6)) - 2):
                if (true_dict[j * 5 + 1][2:].isnumeric() & true_dict[j * 5 + 2][2:].isnumeric() & true_dict[j * 5 + 3][2:].isnumeric()
                & true_dict[j * 5 + 4][2:].isnumeric() & inp_dict[j * 5 + 1][2:].isnumeric() & inp_dict[j * 5 + 2][2:].isnumeric()
                & inp_dict[j * 5 + 3][2:].isnumeric() & inp_dict[j * 5 + 4][2:].isnumeric() & pred_dict[j * 6 + 1][2:].isnumeric() &
                pred_dict[j * 6 + 2][2:].isnumeric() & pred_dict[j * 6 + 3][2:].isnumeric() & pred_dict[j * 6 + 4][2:].isnumeric() ):
                    orig_elt = true_dict[j * 5 + 0]
                    orig_x = int(true_dict[j * 5 + 1][2:])
                    orig_y = int(true_dict[j * 5 + 2][2:])
                    orig_h = int(true_dict[j * 5 + 3][2:])
                    orig_w = int(true_dict[j * 5 + 4][2:])
                    #orig_grp = int(true_dict[j * 6 + 5][2:])
                    # if len(gt_data['categories'])==0:
                    #     gt_data['categories'].append([{
                    #         'id': t[j * 5 + 0],
                    #         'supercategory': 'none',
                    #         'name': true_dict[j * 5 + 0]
                    #     }])
                    #     print(gt_data['categories'])
                    # else:
                    #     cat_check = next((item for item in gt_data['categories'] if item[0]["id"] == t[j*5+0]), False)
                    #     if cat_check==False:
                    #         gt_data['categories'].append([{
                    #             'id': t[j*5+0],
                    #             'supercategory': 'none',
                    #             'name': true_dict[j * 5 + 0]
                    #         }])


                    inp_elt = inp_dict[j * 5 + 0]
                    inp_x = int(inp_dict[j * 5 + 1][2:])
                    inp_y = int(inp_dict[j * 5 + 2][2:])
                    inp_h = int(inp_dict[j * 5 + 3][2:])
                    inp_w = int(inp_dict[j * 5 + 4][2:])

                    pred_elt = pred_dict[j * 6 + 0]
                    pred_x = int(pred_dict[j * 6 + 1][2:])
                    pred_y = int(pred_dict[j * 6 + 2][2:])
                    pred_h = int(pred_dict[j * 6 + 3][2:])
                    pred_w = int(pred_dict[j * 6 + 4][2:])

                    pred_df.loc[len(pred_df.index)] = [filename, pred_x, pred_y, pred_h, pred_w]
                    # row1 = pd.DataFrame({"filename": filename,
                    #               "x1": pred_x,
                    #               "y1":pred_y,
                    #               "ht":pred_h,
                    #                "wd":pred_w})
                    # pred_df = pred_df.append(row1, ignore_index=True)


                    # gt_data['annotations'].append({
                    # 'id': ann_id,
                    # 'image_id': i,
                    # 'category_id': t[j * 5 + 0],
                    # 'bbox': [orig_x,orig_y, orig_w, orig_h],
                    # 'area': orig_w*orig_h,
                    # 'iscrowd':0
                    # })
                    # ann_id=ann_id+1
                    # op_data.append({
                    #     'image_id': i,
                    #     'category_id': p[j * 6 + 0],
                    #     'bbox': [pred_x, pred_y, pred_w, pred_h],
                    #     'score': 1
                    # })
                    #
                    # ip_data.append({
                    #     'image_id': i,
                    #     'category_id': inp[j * 5 + 0],
                    #     'bbox': [inp_x, inp_y, inp_w, inp_h],
                    #     'score': 1
                    # })

                    #Writing to files for first coco api
                    # \n is placed to indicate EOL (End of Line)
                    ip_file.write(str(inp_elt) + " 1 " + str(inp_x) + ' ' + str(inp_y) + " " + str(inp_x + inp_w)
                                  + " " + str(inp_y + inp_h) + "\n")
                    gt_file.write( str(orig_elt)+" " +str(orig_x) +' ' + str(orig_y) +" "+ str(orig_x+orig_w)+" "
                                   +str(orig_y+orig_h)+"\n")
                    op_file.write(str(pred_elt) + " 1 " + str(pred_x) + ' ' + str(pred_y) + " " + str(pred_x + pred_w)
                                  + " " + str(pred_y + pred_h) + "\n")


            #         if pred_dict[j * 6 + 5][2:] in colors.keys():
            #             pred_grp = int(pred_dict[j * 6 + 5][2:])
            #
            #         else:
            #             break
            #
            #         ax1.imshow(elementdict[inp_elt], extent=[orig_x, orig_w + orig_x, orig_y, orig_h + orig_y])
            #         # ax1.add_patch(Rectangle((inp_x+3, inp_y+3), inp_w, inp_h, facecolor=(0, 0, 0, 0)))
            #         ax1.add_patch(matplotlib.patches.Rectangle((orig_x, orig_y),
            #                                                    orig_w, orig_h,
            #                                                    color='black', fill=0))
            #
            #         ax2.imshow(elementdict[inp_elt], extent=[inp_x, inp_w+inp_x, inp_y, inp_h+inp_y])
            #         #ax1.add_patch(Rectangle((inp_x+3, inp_y+3), inp_w, inp_h, facecolor=(0, 0, 0, 0)))
            #         ax2.add_patch(matplotlib.patches.Rectangle((inp_x, inp_y),
            #                                              inp_w, inp_h,
            #                                              color='black', fill=0))
            #         ax3.imshow(elementdict[inp_elt], extent=[pred_x, pred_w + pred_x, pred_y, pred_h + pred_y])
            #
            #         ax3.add_patch(matplotlib.patches.Rectangle((pred_x, pred_y),
            #                                              pred_w, pred_h,
            #                                              color=colors[str(pred_grp)], fill=0))
            #
            # ax1.set_xlim([-1, 10])
            # ax1.set_ylim([-1, 16])
            # ax2.set_xlim([-1, 10])
            # ax2.set_ylim([-1, 16])
            # ax3.set_xlim([-1, 10])
            # ax3.set_ylim([-1, 16])

            #plot_to_tensorboard(writer, filename, fig, 0)

            #
            # plt.show()
            ip_file.close()
            op_file.close()
            gt_file.close()
    # with open('op_data.json', 'w') as outfile:
    #     json.dump(op_data, outfile)
    # with open('ip_data.json', 'w') as outfile:
    #     json.dump(ip_data, outfile)
    # with open('gt_data.json', 'w') as outfile:
    #     json.dump(gt_data, outfile)

    return pred_df


def plot(i_list, p_list, t_list, dataset, orig_data):
    for i in range(len(i_list)):
        inp = i_list[i]
        p = p_list[i]
        t = t_list[i]
        if(1 in inp):
            index_1 = inp.index(1)
            inp = inp[:index_1]
            p = p[:index_1]
            t = t[:index_1]
        inp_dict=get_orignal_data(dataset, inp)
        pred_dict=get_orignal_data(dataset, p)
        true_dict=get_orignal_data(dataset, t)
        print(pred_dict)
        k=' '.join(inp_dict)
        #print(k)
        #print(orig_data[orig_data['seq_x']==k]['UIID'].values[0])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        seq_x = ' '.join(inp_dict)
        #fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        for j in range(0,len(inp),5):
            orig_elt=true_dict[j+0]
            orig_x=int(true_dict[j+1][2:])
            orig_y=int(true_dict[j+2][2:])
            orig_h=int(true_dict[j+3][2:])
            orig_w=int(true_dict[j+4][2:])

            inp_elt = inp_dict[j + 0]
            inp_x = int(inp_dict[j + 1][2:])
            inp_y = int(inp_dict[j + 2][2:])
            inp_h = int(inp_dict[j + 3][2:])
            inp_w = int(inp_dict[j + 4][2:])

            pred_elt = pred_dict[j + 0]
            pred_x = int(pred_dict[j + 1][2:])
            pred_y = int(pred_dict[j + 2][2:])
            pred_h = int(pred_dict[j + 3][2:])
            pred_w = int(pred_dict[j + 4][2:])

            if((pred_x!='<unk>' )& (pred_y!='<unk>') & (pred_h!='<unk>') & (pred_w!='<unk>') &
                (inp_x!='<unk>' )& (inp_y!='<unk>') & (inp_h!='<unk>') & (inp_w!='<unk>') &
                (orig_x!='<unk>' )& (orig_y!='<unk>') & (orig_h!='<unk>') & (orig_w!='<unk>')):
                rect1 = matplotlib.patches.Rectangle((orig_x, orig_y),
                                                     orig_w, orig_h,
                                                     color='black', fill=0)
                ax1.add_patch(rect1)
                ax1.annotate(orig_elt, (orig_x+0.2, orig_y+0.2),
                             fontsize=6, ha='left', va='bottom')

                rect2 = matplotlib.patches.Rectangle((inp_x, inp_y),
                                                     inp_w, inp_h,
                                                     color='red', fill=0)
                ax2.add_patch(rect2)
                ax2.annotate(orig_elt, (inp_x+0.2, inp_y+0.2), color='red',
                             fontsize=6, ha='left', va='bottom')

                rect3 = matplotlib.patches.Rectangle((pred_x, pred_y),
                                                     pred_w, pred_h,
                                                     color='green', fill=0)
                ax3.add_patch(rect3)
                ax3.annotate(orig_elt, (pred_x+0.2, pred_y+0.2), color='green',
                            fontsize=6, ha='left', va='bottom')

        ax1.set_xlim([-1, 18])
        ax1.set_ylim([-1, 32])
        ax2.set_xlim([-1, 18])
        ax2.set_ylim([-1, 32])
        ax3.set_xlim([-1, 18])
        ax3.set_ylim([-1, 32])
        plt.show()


def plot_with_group(i_list, p_list, t_list, dataset, orig_data):
    colors = {
        "1": "blue",
        "2": "red",
        "3": "green",
        "4": "yellow",
        "5": "grey",
        "6": "black",
        "7": "cyan",
        "8": "brown"
    }
    for i in range(len(i_list)):
        inp = i_list[i]
        p = p_list[i]
        t = t_list[i]
        if(1 in inp):
            index_1 = inp.index(1)
            inp = inp[:index_1]
            p = p[:index_1]
            t = t[:index_1]
        inp_dict=get_orignal_data(dataset, inp)
        pred_dict=get_orignal_data(dataset, p)
        true_dict=get_orignal_data(dataset, t)
        print(true_dict)
        print(pred_dict)
        k=' '.join(inp_dict)
        #print(k)
        #print(orig_data[orig_data['seq_x']==k]['UIID'].values[0])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        seq_x = ' '.join(inp_dict)
        fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')

        for j in range(0,int(np.floor(len(inp)/5))-2):
            orig_elt=true_dict[j*6+0]
            orig_x=int(true_dict[j*6+1][2:])
            orig_y=int(true_dict[j*6+2][2:])
            orig_h=int(true_dict[j*6+3][2:])
            orig_w=int(true_dict[j*6+4][2:])
            orig_grp = int(true_dict[j * 6 + 5][2:])

            inp_elt = inp_dict[j*5 + 0]
            inp_x = int(inp_dict[j*5 + 1][2:])
            inp_y = int(inp_dict[j*5 + 2][2:])
            inp_h = int(inp_dict[j*5 + 3][2:])
            inp_w = int(inp_dict[j*5 + 4][2:])

            pred_elt = pred_dict[j*6 + 0]
            pred_x = int(pred_dict[j*6 + 1][2:])
            pred_y = int(pred_dict[j*6 + 2][2:])
            pred_h = int(pred_dict[j*6 + 3][2:])
            pred_w = int(pred_dict[j*6 + 4][2:])
            pred_grp = int(pred_dict[j*6 + 5][2:])

            rect1 = matplotlib.patches.Rectangle((orig_x, orig_y),
                                                 orig_w, orig_h,
                                                 color=colors[str(orig_grp)], fill=0)
            ax1.add_patch(rect1)
            ax1.annotate(orig_elt, (orig_x+0.2, orig_y+0.2),
                         fontsize=6, ha='left', va='bottom')

            rect2 = matplotlib.patches.Rectangle((inp_x, inp_y),
                                                 inp_w, inp_h,
                                                 color='red', fill=0)
            ax2.add_patch(rect2)
            ax2.annotate(orig_elt, (inp_x+0.2, inp_y+0.2), color='red',
                         fontsize=6, ha='left', va='bottom')

            rect3 = matplotlib.patches.Rectangle((pred_x, pred_y),
                                                 pred_w, pred_h,
                                                 color=colors[str(pred_grp)], fill=0)
            ax3.add_patch(rect3)
            ax3.annotate(orig_elt, (pred_x+0.2, pred_y+0.2), color=colors[str(pred_grp)],
                        fontsize=6, ha='left', va='bottom')

        ax1.set_xlim([-1, 18])
        ax1.set_ylim([-1, 32])
        ax2.set_xlim([-1, 18])
        ax2.set_ylim([-1, 32])
        ax3.set_xlim([-1, 18])
        ax3.set_ylim([-1, 32])
        plt.show()


def testing(test_iter, model, vocab_size):
    model.eval()
    preds = []
    label = []
    softmax = nn.Softmax(dim=-1)
    i_list = []
    p_list = []
    t_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            # generate random num between 0 and bs(32)
            #randnum = random.randint(0, 31)
            input_ids = batch.seq_x.cpu()
            truelabel_cls = batch.seq_y.transpose(0,1).cpu()
            mask = create_masks(batch)
            att_mask = mask.cpu()
            logits_cls = model(input_ids, att_mask)
            prediction = softmax(logits_cls).argmax(2)
            inp = input_ids.transpose(0,1)
            print(inp.size())
            if(inp.size()[0]==32):          #batch size
                 #i = inp[randnum].cpu().detach().numpy().tolist()
                 #p = prediction[randnum].cpu().detach().numpy().tolist()
                 #t = truelabel_cls[randnum].cpu().detach().numpy().tolist()
                 for i in range(32):
                     input = inp[i].cpu().detach().numpy().tolist()
                     p = prediction[i].cpu().detach().numpy().tolist()
                     t = truelabel_cls[i].cpu().detach().numpy().tolist()
                     if (1 in input):
                        index_1 = input.index(1)
                        #if(index_1 < 70):      #plot those with less than 10 elements
                        i_list.append(input)
                        p_list.append(p)
                        t_list.append(t)

            nptrue, nppreds = prune_preds(truelabel_cls.contiguous().view(-1), prediction.view(-1))
            label.extend(nptrue)
            preds.extend(nppreds)

    return label, preds, i_list, p_list, t_list






def main():
    # device = torch.device("cuda:1")
    # torch.cuda.set_device(device)
    model_path = 'results/model.tar'
    print(model_path)
    dataset = UIdataset()
    vocab_size = dataset.get_vocab_size()
    test_iter = dataset.add_new_test_file(path='../data/processed/Test_data_with_prefix.csv')
    transformer = torch.load(model_path)
    #print(len(test_iter))
    label, preds, i , p, t = testing(test_iter, transformer, vocab_size)
    f1, align_acc = f1score_without_group(label, preds, average='weighted')

    #print('test_acc:', acc)
    print('test_f1:', f1)
    #print('group accuracy: ', group_acc)
    print('alignment accuracy:', align_acc)
    orig_data = pd.read_csv("../data/processed/SynZ_data_with_prefix.csv", header=0)
    pred_df = plot_images(i, p, t, dataset, orig_data)



if __name__ == "__main__":
    main()
