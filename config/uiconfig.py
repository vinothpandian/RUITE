import json
# Opening JSON file
import os

print(os.getcwd())
f = open('../config/config.json', )
config = json.load(f)
print(config)

class UIconfig:

    seq_len = config['seq_len']
    batch_size = config['batch_size']
    seed = config['seed']
    ignore_index = config['ignore_index']  #padding - dont calculate attention
    lr_main = config['lr_main']
    drop_out = config['drop_out']
    epochs = config['epochs']


    #transformer params
    nhead = config['nhead']
    num_encoder_layers = config['num_encoder_layers']
    d_model = config['d_model']

    #discretize
    x_boxes=config['x_boxes']
    y_boxes=config['y_boxes']

    #include grouping data in training?
    group=config['group']

    #include element names in sequence
    elt_name_include=config['elt_name_include']

    # noise addidtion
    noise_dist = config['noise_dist']  #can be 'uniform', 'normal', 'triangular'
    noise_stddev = config['noise_stddev']     #for normal

    noise_low=config['noise_low']      #for triangular or uniform
    noise_high=config['noise_high']

    #for calculating mAP
    minoverlap = config['minoverlap']

    filename = config['filename']
    #path to save model at:
    model_path= config['model_path']
