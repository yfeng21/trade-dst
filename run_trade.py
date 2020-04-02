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
    parser.add_argument('--gradient_accum_steps', type=int, default=1)
    parser.add_argument('--save_path')

    parser.add_argument('--all_vocab', type=bool, default=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--cache_path', type=str, default='./cache')
    return parser.parse_args()


def start_train(args, model, train, dev, slots):
    curr_acc, best_acc = 0.0, 48.0
    best_acc = model.evaluate(dev, best_acc, slots, None)
    for it in range(args.epoch):
        progress_bar = tqdm(enumerate(train),total=len(train))
        for i,d in progress_bar:
            loss = model.run_batch(d, slots, reset=(i==0))
            if args.gradient_accum_steps > 1:
                loss = loss / args.gradient_accum_steps
            loss.backward()
            # model.optimize(1.0)
            if (i+1) % args.gradient_accum_steps == 0:
                model.clip(1.0)
                model.optimizer.step()
                model.optimizer.zero_grad()
            progress_bar.set_description(model.print_loss())

        curr_acc = model.evaluate(dev, best_acc, slots, None)
        model.scheduler.step(curr_acc)
        if curr_acc >= best_acc:
            best_acc = curr_acc
            best_model = model


def start_test(model, test, slot_test):
    model.evaluate(test, 1e7, slot_test)


def main():
    args = parse_argument()
    train, dev, test, lang, ALL_SLOTS, gating_dict = prepare_data_seq(args)
    model = TRADE(
        args.hidden_size,
        lang,
        args.save_path,
        args.lr if args.train else 0,
        args.drop_rate if args.train else 0,
        ALL_SLOTS,
        gating_dict,
        emb_path=args.embedding
    )
    if args.train:
        start_train(args, model, train, test, ALL_SLOTS)
    else:
        start_test(model, test, ALL_SLOTS)


if __name__ == "__main__":
    main()
