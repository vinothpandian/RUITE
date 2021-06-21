import random
import sys

import numpy as np
import torch.optim as optim
import tqdm
from config.uiconfig import *
from models.dataset import UIdataset
from models.model import *
from Utils.utils import *


def training(valid_iter, batch_iter, writer, model, optim,schedular, criterion, train_iter, epoch, vocab_size):

    print(len(train_iter), len(valid_iter))
    sys.exit()
    model.train()
    losses = []
    label = []
    preds = []
    softmax = nn.Softmax(dim = -1)
    print('\nTrain_Epoch:', epoch)
    for batch in tqdm.tqdm(train_iter):
        batch_iter=batch_iter+1
        optim.zero_grad()
        mask = create_masks(batch)
        input = batch.seq_x.cpu()
        truelabel_cls = batch.seq_y.transpose(0,1).cpu()
        attn_mask = mask.cpu()

        logits_cls = model(input, attn_mask)

        loss_cls = criterion(logits_cls.view(-1, vocab_size), truelabel_cls.contiguous().view(-1, ))

        loss = loss_cls

        if batch_iter%50 == 0:
            valid_loss, valid_label, valid_preds = validation(model, criterion, valid_iter, epoch, vocab_size)
            writer.add_scalars('Stepwise cross-entropy loss', {'training ': loss,
                                                                'validation' : sum(valid_loss) / len(valid_loss)},
                                                                batch_iter)
        losses.append(loss.item())

        #for now we are only interested in accuracy and f1 of the classification task
        preds_cls = softmax(logits_cls).argmax(2)   #to get a vector of probabilties for each word

        #append labels & preds of each batch to calculate score for every epoch:
        nptrue, nppreds = prune_preds(truelabel_cls.contiguous() .view(-1), preds_cls.view(-1))

        label.extend(nptrue)
        preds.extend(nppreds)

        loss.backward()

        optim.step()

        if schedular is not None:
            schedular.step()

    return losses, label, preds, batch_iter

def validation(model, criterion, valid_iter, epoch, vocab_size):
    model.eval()
    losses = []
    label = []
    preds = []

    softmax = nn.Softmax(dim=-1)
    print('\nValid_Epoch:', epoch)

    with torch.no_grad():
        for batch in tqdm.tqdm(valid_iter):

            mask = create_masks(batch)
            input = batch.seq_x.cpu()
            truelabel_cls = batch.seq_y.transpose(0,1).cpu()
            attn_mask = mask.cpu()
            logits_cls = model(input, attn_mask)
            loss_cls = criterion(logits_cls.view(-1, vocab_size), truelabel_cls.contiguous().view(-1, ))
            loss = loss_cls

            losses.append(loss.item())

            preds_cls = softmax(logits_cls).argmax(2)

            #plot_save(truelabel_cls, preds_cls, epoch)

            nptrue, nppreds = prune_preds(truelabel_cls.contiguous().view(-1), preds_cls.view(-1))

            label.extend(nptrue)
            preds.extend(nppreds)


    return losses, label, preds,

def train_val(writer,train_iter, valid_iter, vocab_size, model_path:str, trial=None, best_params=None):

    epochs = UIconfig.epochs
    lrmain = UIconfig.lr_main
    drop_out = UIconfig.drop_out
    print({'lrmain':lrmain, 'drop_out':drop_out})

    model = Transformer(vocab_size, UIconfig.d_model, UIconfig.num_encoder_layers, UIconfig.nhead, drop_out)
    # this is how they initialize params in paper
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = optim.Adam(model.parameters(), lr=lrmain)

    criterion = nn.CrossEntropyLoss(ignore_index= UIconfig.ignore_index)
    model.cpu()
    score = score_cal()
    batch_iter=1
    for epoch in range(epochs):
        train_losses, label, preds, batch_iter = training(valid_iter, batch_iter, writer, model, optimizer, None, criterion, train_iter, epoch, vocab_size)
        f1, acc, group_acc, align_acc = f1score_with_group(label, preds, 'weighted')
        score.train_f1.append(f1)
        score.train_acc.append(acc)
        score.train_loss.append(sum(train_losses)/len(train_losses))
        print('train_weighted_f1', f1)
        print('train_acc', acc)
        print('train grouping acc:', group_acc)
        print('train alignment acc:', align_acc)

        #Writing to tensorboard
        writer.add_scalar('train_weighted_f1', f1, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar('train grouping acc:', group_acc, epoch)
        writer.add_scalar('train alignment acc:', align_acc, epoch)

        valid_loss, valid_label, valid_preds = validation(model, criterion, valid_iter, epoch, vocab_size)
        valid_f1, valid_acc, group_acc, align_acc = f1score_with_group(valid_label, valid_preds, 'weighted')
        score.valid_f1.append(valid_f1)
        score.valid_acc.append(valid_acc)
        score.valid_loss.append(sum(valid_loss) / len(valid_loss))

        print('valid_weighted_f1:', valid_f1)
        print('valid_acc:', valid_acc)
        print('valid grouping acc:', group_acc)
        print('valid alignment acc:', align_acc)
        #classificationreport(valid_label, valid_preds)

        #Writing to tensorboard
        writer.add_scalar('valid_weighted_f1:', valid_f1, epoch)
        writer.add_scalar('valid_acc:', valid_acc, epoch)
        writer.add_scalar('valid grouping acc:', group_acc, epoch)
        writer.add_scalar('valid alignment acc:', align_acc, epoch)

    if(trial is None):
        print('-saving model-')
        print(model_path)
        #mask = create_masks(batch)
        #input = batch.seq_x.cpu()
        b = next(iter(train_iter))
        mask = create_masks(b).cpu()
        input = b.seq_x.cpu()
        torch.save(model, model_path)
        writer.add_graph(model,(input,mask))

    return score



def main():
    print('hello')
    model_path = '../results/model.tar'

    # device = torch.device("cuda:1")
    # torch.cuda.set_device(device)
    np.random.seed(UIconfig.seed)
    torch.manual_seed(UIconfig.seed)
    random.seed(UIconfig.seed)


    dataset = UIdataset()

    train_iter, valid_iter, test_iter = dataset.get_loaders()

    score = train_val(train_iter, valid_iter, dataset.get_vocab_size(), model_path, None, None)
    print_result(score, UIconfig.epochs)



if __name__ == "__main__":
    main()
