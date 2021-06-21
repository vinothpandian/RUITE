from config.uiconfig import *
from MaP_evaluation.run_eval import *
from models.train import *
from Preprocessing.add_noise import *
from test.test1 import *
from Utils.utils import *
from torch.utils.tensorboard import SummaryWriter


def main():
    writer = SummaryWriter('tensorboard_runs/'+UIconfig.filename)
    # Test data - Adding noise
    # data, xmax, ymax = alter_scale_data("test", x_divide_by=(1442 / UIconfig.x_boxes),
    #                                     y_divide_by=(2806 / UIconfig.y_boxes))
    # noisy_data = add_noise(data, xmax, ymax)
    # fdata = data_format("test",noisy_data)
    # fdata.to_csv("../data/processed/Test_data_with_prefix.csv")

    # # SynZ data - Adding noise
    # data, xmax, ymax = alter_scale_data("train", x_divide_by=(1439 / UIconfig.x_boxes),
    #                                     y_divide_by=(2559 / UIconfig.y_boxes))
    # noisy_data = add_noise(data, xmax, ymax)
    # fdata = data_format("train",noisy_data)
    # fdata.to_csv("../data/processed/SynZ_data_with_prefix.csv")

    #Training...
    print('Training starts..')
    model_path = UIconfig.model_path

    # device = torch.device("cuda:1")
    # torch.cuda.set_device(device)
    np.random.seed(UIconfig.seed)
    torch.manual_seed(UIconfig.seed)
    random.seed(UIconfig.seed)

    dataset = UIdataset()

    train_iter, valid_iter, test_iter = dataset.get_loaders()
    score = train_val(writer,train_iter, valid_iter, dataset.get_vocab_size(), model_path, None, None)
    print_result(score, UIconfig.epochs)
    print('Training ends.')

    print('Testing..')
    # device = torch.device("cuda:1")
    # torch.cuda.set_device(device)
    model_path = UIconfig.model_path
    dataset = UIdataset()
    vocab_size = dataset.get_vocab_size()
    test_iter = dataset.add_new_test_file(path='../data/processed/Test_data_with_prefix.csv')
    transformer = torch.load(model_path)
    # print(len(test_iter))
    label, preds, i, p, t = testing(test_iter, transformer, vocab_size)
    f1, align_acc = f1score_without_group(label, preds, average='weighted')

    # print('test_acc:', acc)
    #writer.add_text('test_f1: ', f1.astype(str))
    #writer.add_text('test_accuracy: ', align_acc.astype(str))


    print('test_f1:', f1)
    # print('group accuracy: ', group_acc)
    print('alignment accuracy:', align_acc)
    orig_data = pd.read_csv("../data/processed/SynZ_data_with_prefix.csv", header=0)
    pred_df = plot_images(writer, i, p, t, dataset, orig_data)
    gest_align = gestalt_alignment(pred_df)
    writer.add_text('Gestalt_alignment_score: ', str(gest_align))

    MINOVERLAP = 0.5
    MAP_eval(MINOVERLAP, writer, 1)
    MINOVERLAP = 0.6
    MAP_eval(MINOVERLAP, writer, 1)
    MINOVERLAP = 0.7
    MAP_eval(MINOVERLAP, writer, 1)
    MINOVERLAP = 0.8
    MAP_eval(MINOVERLAP, writer, 1)
    MINOVERLAP = 0.9
    MAP_eval(MINOVERLAP, writer, 1)
    MINOVERLAP = 1
    MAP_eval(MINOVERLAP, writer, 1)
    #MAP_eval(writer, 0)

    writer.flush()

if __name__ == "__main__":
    main()
