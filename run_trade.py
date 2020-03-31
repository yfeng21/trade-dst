import argparse
from tqdm import tqdm
from models.TRADE import *
from utils.utils_multiWOZ_DST import *


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',help="training mode",action="store_true")
    parser.add_argument('--data_dir',help="path to processed data directory",required=True)
    parser.add_argument('--embedding',help="path to pre-trained embedding file")
    parser.add_argument('--epoch',default=10,type=int)
    parser.add_argument('--drop_rate', type=float)
    parser.add_argument('--hidden_size', type=int, default=400)
    # parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--save_path')

    parser.add_argument('--all_vocab', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--cache_path', type=str, default='./cache')
    return parser.parse_args()


def start_train(args, model, train, dev, slot_train, slot_dev):
    curr_acc, best_acc = 0.0, 48.0
    for it in range(args.epoch):
        progress_bar = tqdm(enumerate(train),total=len(train))
        for i,d in progress_bar:
            model.train_batch(d, 1, slot_train, reset=(i==0))
            # model.optimize(1.0)
            progress_bar.set_description(model.print_loss())
        curr_acc = model.evaluate(dev, best_acc, slot_dev, None)
        model.scheduler.step(curr_acc)
        if curr_acc >= best_acc:
            best_acc = curr_acc
            best_model = model


def start_test(model, test, slot_test):
    model.evaluate(test, 1e7, slot_test)


def main():
    args = parse_argument()
    train, dev, test, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(args)
    model = TRADE(
        args.hidden_size,
        lang,
        args.save_path,
        "dst",
        args.lr if args.train else 0,
        args.drop_rate if args.train else 0,
        SLOTS_LIST,
        gating_dict,
        emb_path=args.embedding
    )
    if args.train:
        #start_train(args, model, train, test, SLOTS_LIST[1], SLOTS_LIST[3])
        start_train(args, model, test, test, SLOTS_LIST[3], SLOTS_LIST[3])
    else:
        start_test(model, test, SLOTS_LIST[3])


if __name__ == "__main__":
    main()
