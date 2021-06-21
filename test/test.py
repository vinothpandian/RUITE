import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from models.dataset import UIdataset
from Utils.utils import *


def get_orignal_data(dataset, inp):
    l = []
    for i in inp:
        l.append(dataset.get_vocab_item(i))
    return l

def plot_images(i_list, p_list, t_list, dataset, orig_data):
    #Reading all elements and saving in dict
    # elementdict={
    # "alert":read_png('../data/images/alert.png'),
    # "button":read_png('../data/images/button.png'),
    # "card":read_png('../data/images/card.png'),
    # "checkbox_checked":read_png('../data/images/checkbox_checked.png'),
    # "checkbox_unchecked":read_png('../data/images/checkbox_unchecked.png'),
    # "chip":read_png('../data/images/chip.png'),
    # "data_table":read_png('../data/images/data_table.png'),
    # "floating_action_button":read_png('../data/images/floating_action_button.png'),
    # "grid_list":read_png('../data/images/grid_list.png'),
    # "image":read_png('../data/images/image.png'),
    # "label":read_png('../data/images/label.png'),
    # "menu":read_png('../data/images/menu.png'),
    # "radio_button_checked":read_png('../data/images/radio_button_checked.png'),
    # "radio_button_unchecked":read_png('../data/images/radio_button_unchecked.png'),
    # "slider":read_png('../data/images/slider.png'),
    # "switch_disabled":read_png('../data/images/switch_disabled.png'),
    # "switch_enabled":read_png('../data/images/switch_enabled.png'),
    # "text_area":read_png('../data/images/text_area.png'),
    # "text_field":read_png('../data/images/text_field.png'),
    # "tooltip":read_png('../data/images/tooltip.png')
    # }

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
        fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
        ax1.axes.xaxis.set_visible(False)
        ax2.axes.xaxis.set_visible(False)
        ax3.axes.xaxis.set_visible(False)

        ax1.axes.yaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax3.axes.yaxis.set_visible(False)

        for j in range(0, int(np.floor(len(inp) / 6)) - 2):
            orig_elt = true_dict[j * 6 + 0]
            orig_x = int(true_dict[j * 6 + 1][2:])
            orig_y = int(true_dict[j * 6 + 2][2:])
            orig_h = int(true_dict[j * 6 + 3][2:])
            orig_w = int(true_dict[j * 6 + 4][2:])
            orig_grp = int(true_dict[j * 6 + 5][2:])

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
            pred_grp = int(pred_dict[j * 6 + 5][2:])

            ax1.imshow(elementdict[inp_elt], extent=[orig_x, orig_w + orig_x, orig_y, orig_h + orig_y])
            # ax1.add_patch(Rectangle((inp_x+3, inp_y+3), inp_w, inp_h, facecolor=(0, 0, 0, 0)))
            ax1.add_patch(matplotlib.patches.Rectangle((orig_x, orig_y),
                                                       orig_w, orig_h,
                                                       color=colors[str(orig_grp)], fill=0))

            ax2.imshow(elementdict[inp_elt], extent=[inp_x, inp_w+inp_x, inp_y, inp_h+inp_y])
            #ax1.add_patch(Rectangle((inp_x+3, inp_y+3), inp_w, inp_h, facecolor=(0, 0, 0, 0)))
            ax2.add_patch(matplotlib.patches.Rectangle((inp_x, inp_y),
                                                 inp_w, inp_h,
                                                 color='black', fill=0))
            ax3.imshow(elementdict[inp_elt], extent=[pred_x, pred_w + pred_x, pred_y, pred_h + pred_y])
            ax3.add_patch(matplotlib.patches.Rectangle((pred_x, pred_y),
                                                 pred_w, pred_h,
                                                 color=colors[str(pred_grp)], fill=0))

        ax1.set_xlim([-1, 18])
        ax1.set_ylim([-1, 32])
        ax2.set_xlim([-1, 18])
        ax2.set_ylim([-1, 32])
        ax3.set_xlim([-1, 18])
        ax3.set_ylim([-1, 32])
        plt.show()


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
        fig.suptitle(orig_data[orig_data['seq_x'] == seq_x]['UIID'].values[0])
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
    model_path = '../results/model.tar'
    print(model_path)
    dataset = UIdataset()
    vocab_size = dataset.get_vocab_size()
    _, _, test_iter = dataset.get_loaders()
    transformer = torch.load(model_path)
    print(len(test_iter))
    label, preds,i , p, t = testing(test_iter, transformer, vocab_size)
    f1, acc, group_acc, align_acc = f1score_with_group(label, preds, average='weighted')

    print('test_acc:', acc)
    print('test_f1:', f1)
    print('group accuracy: ', group_acc)
    print('alignment accuracy:', align_acc)
    orig_data = pd.read_csv("../data/processed/SynZ_data_with_prefix.csv", header=0)
    plot_images(i, p, t, dataset, orig_data)

if __name__ == "__main__":
    main()
