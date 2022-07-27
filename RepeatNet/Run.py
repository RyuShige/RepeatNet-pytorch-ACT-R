import sys
sys.path.append('./')
from RepeatNet.Dataset import *
from torch import optim
from Common.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from RepeatNet.Model import *
import codecs
import numpy as np
import random
import os
from Common.pytorchtools import EarlyStopping

def get_ms():
    return time.time() * 1000

def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

base_output_path = './output/RepeatNet/' # colab上で実行する場合、カレントディレクトリに戻ってしまうので../ではなく./
base_data_path = './datasets/demo/'
dir_path = os.path.dirname(os.path.realpath(__file__)) # ファイルが存在するディレクトリのパスを取得
epoches=100
embedding_size=128
hidden_size=128
item_vocab_size=10+1


def train(args):
    batch_size = 1024

    path = os.getcwd()
    print(path)

    train_dataset = RepeatNetDataset(base_data_path+'demo.train')
    train_size=train_dataset.len

    model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
    init_params(model)

    trainer = CumulativeTrainer(model, None, None, args.local_rank, 4)
    model_optimizer = optim.Adam(model.parameters())

    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(base_output_path, 'model/'))

    for i in range(epoches):
        trainer.train_epoch('train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        loss_epoch = trainer.loss_epoch
        early_stopping(loss_epoch, model, i) # 最良モデルならモデルパラメータ保存
        if early_stopping.early_stop: 
            break
        # trainer.serialize(i, output_path=base_output_path)

def infer(args):
    batch_size = 1024

    valid_dataset = RepeatNetDataset(base_data_path + 'demo.valid')
    test_dataset = RepeatNetDataset(base_data_path + 'demo.test')

    for i in range(epoches):
        print('epoch', i)
        file = base_output_path+'model/'+ str(i) + '.pkl'

        result_output_path = os.path.join(base_output_path, 'result/')
        if not os.path.exists(result_output_path):
            os.makedirs(result_output_path)

        if os.path.exists(file):
            model = RepeatNet(embedding_size, hidden_size, item_vocab_size)
            model.load_state_dict(torch.load(file, map_location='cpu'))
            trainer = CumulativeTrainer(model, None, None, args.local_rank, 4)

            rs = trainer.predict('infer', valid_dataset, collate_fn, batch_size, i, base_output_path)
            file = codecs.open(base_output_path+'result/'+str(i)+'.'+str(args.local_rank)+'.valid', mode='w', encoding='utf-8')
            for data, output in rs:
                scores, index=output
                label=data['item_tgt']
                for j in range(label.size(0)):
                    file.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
            file.close()
            rs = trainer.predict('infer', test_dataset, collate_fn, batch_size, i, base_output_path)
            file = codecs.open(base_output_path + 'result/' + str(i)+'.'+str(args.local_rank)+'.test', mode='w', encoding='utf-8')
            for data, output in rs:
                scores, index = output
                label = data['item_tgt']
                for j in range(label.size(0)):
                    file.write('[' + ','.join([str(id) for id in index[j, :50].tolist()]) + ']|[' + ','.join([str(id) for id in label[j].tolist()]) + ']' + os.linesep)
            file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())
    init_seed(123456)

    if args.mode=='infer':
        infer(args)
    elif args.mode=='train':
        train(args)
