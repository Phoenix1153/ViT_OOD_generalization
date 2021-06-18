import torch
import torch.nn.functional as F
import argparse
import tqdm

from dataset import CustomDataset, CustomDataLoader
from models import deit_small_b16_384



def build_dataset(root_dir, meta_file):
    dataset = CustomDataset(root_dir, meta_file)
    return dataset


def build_dataloader(dataset, batch_size, num_workers):
    dataloader = CustomDataLoader(dataset, batch_size, num_workers)
    return dataloader


def build_model(model, num_classes, checkpoint):
    if model == "deit_small_b16_384":
        model = deit_small_b16_384(num_classes=num_classes)
    data = torch.load(checkpoint, map_location='cpu')['model']
    new_state_dict = {}
    for k, v in data.items():
        new_k = k.replace("module.","")
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=True)
    return model


def main(args):

    dataset = build_dataset(args.root_dir, args.meta_file)
    dataloader = build_dataloader(dataset, args.bs, args.num_workers)
    model = build_model(args.model, args.num_classes, args.checkpoint)
    model = model.cuda()

    pred = torch.zeros(len(dataset)).long()
    gt = torch.zeros(len(dataset)).long()
    max_bs = args.bs
    cur_idx = 0
    model.eval()
    for data in tqdm.tqdm(dataloader):
        image = data['image']
        label = data['label']
        image = image.cuda()
        logits = model(image)
        cur_bs = image.shape[0]
        pred[max_bs * cur_idx: max_bs * cur_idx + cur_bs] = torch.argmax(logits, dim=1).cpu()
        gt[max_bs * cur_idx: max_bs * cur_idx + cur_bs] = label
        cur_idx += 1
    acc = torch.mean((gt == pred).float())
    print('Top1 Accuracy: %.5f' % (acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Tool')
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--num-classes', required=True, type=int)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--meta-file', required=True, type=str)
    parser.add_argument('--root-dir', required=True, type=str)
    parser.add_argument('--bs', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    args = parser.parse_args()
    main(args)