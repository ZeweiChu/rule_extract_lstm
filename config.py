import argparse

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Basics

    parser.add_argument('--use_cuda',
                        type=int,
                        default=0,
                        help='use cuda GPU or not')
    parser.add_argument('--decompose_type',
                        type=str,
                        default="beta",
                        help='beta|gamma for decompose type')

    parser.add_argument('--model_file',
                        type=str,
                        default="model.th",
                        help='model file')

    parser.add_argument('--model',
                        type=str,
                        default="LSTMModel",
                        help='choose the model')

    parser.add_argument('--criterion',
                        type=str,
                        default="",
                        help='choose the loss criterion')



    # Data file
    parser.add_argument('--train_file',
                        type=str,
                        default="data/yelp_academic_dataset_review_train.json",
                        help='file containing training data')

    parser.add_argument('--dev_file',
                        type=str,
                        default="data/yelp_academic_dataset_review_dev.json",
                        help='file containing dev data')

    parser.add_argument('--test_file',
                        type=str,
                        default="data/yelp_academic_dataset_review_test.json",
                        help='file containing test data')

    parser.add_argument('--heatmap_file',
                        type=str,
                        default="heatmap.html",
                        help='heat map file')

    parser.add_argument('--vocab_file',
                        type=str,
                        default="data/bobsue.voc.txt",
                        help='dictionary file')

    parser.add_argument('--embedding_file',
                        type=str,
                        default=None,
                        help='Word embedding file')

    parser.add_argument('--max_train',
                        type=int,
                        default=10000,
                        help='Maximum number of dev examples to evaluate on')

    parser.add_argument('--max_dev',
                        type=int,
                        default=1000,
                        help='Maximum number of dev examples to evaluate on')

    parser.add_argument('--vocab_size', 
                        type=int,
                        default=50000,
                        help='maximum number of vocabulary')

    parser.add_argument('--relabeling',
                        type='bool',
                        default=True,
                        help='Whether to relabel the entities when loading the data')

    # Model details
    parser.add_argument('--embedding_size',
                        type=int,
                        default=300,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')

    parser.add_argument('--bidir',
                        type='bool',
                        default=True,
                        help='bidir: whether to use a bidirectional RNN')

    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of RNN layers')

    parser.add_argument('--rnn_type',
                        type=str,
                        default='gru',
                        help='RNN type: lstm or gru (default)')

    parser.add_argument('--att_func',
                        type=str,
                        default='bilinear',
                        help='Attention function: bilinear (default) or mlp or avg or last or dot')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs')

    parser.add_argument('--eval_iterations',
                        type=int,
                        default=10,
                        help='Evaluation on dev set after K epochs')

    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.2,
                        help='Dropout rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        help='Optimizer: sgd or adam (default) or rmsprop')

    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for SGD')

    parser.add_argument('--grad_clipping',
                        type=float,
                        default=1,
                        help='Gradient clipping')

    parser.add_argument('--num_sampled',
                        type=int,
                        default=10,
                        help='number of negative sampling')

    parser.add_argument('--start_from',
                        type=str,
                        default=None,
                        help='start training from a checkpoint')
    parser.add_argument('--load_best',
                        type=int,
                        default=0,
                        help='load best model or not')
    parser.add_argument('--attention_type',
                        type=str,
                        default="dot",
                        help='type of the attention layer: dot/bilinear/concat')

    return parser.parse_args()
