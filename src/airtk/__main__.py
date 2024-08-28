# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
from argparse import ArgumentParser, Namespace
from typing import Final

from airtk.defense import (
    AdtTraining,
    AdtppTraining,
    CurratTraining,
    FatTraining,
    FeatureScatterTraining,
    GairatTraining,
    TradesTraining,
    TradesawpTraining,
    VaTraining,
    YopoTraining,
)


def get_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser("AirTk CLI Interface", description="Run adversarial training experiments.")
    parser.add_argument("training-method", type=str, choices=["adt", "adtpp", "currat", "fat", "fs", "gairat", "trades", "tradesawp", "va", "yopo"], required=True)
    parser.add_argument("dataset", type=str, required=True)
    parser.add_argument("model", type=str, required=True)
    parser.add_argument("out-dir", type=str)
    parser.add_argument("")
    return parser.parse_args()


def main() -> None:
    args: Final[Namespace] = get_args()
    if args.training_method == "adt":
        AdtTraining(args.dataset, args.model, model_dir=args.out_dir)()
    elif args.training_method == "adtpp":
        AdtppTraining(args.dataset, args.model, model_dir=args.out_dir)()
    elif args.training_method == "currat":
        CurratTraining(args.dataset, args.model, model_dir=args.out_dir)()
    elif args.training_method == "fat":
        FatTraining(args.dataset, args.model, out_dir=args.out_dir)()
    elif args.training_method == "fs":
        FeatureScatterTraining(args.dataset, 10 if args.dataset in {"cifar10", "mnist"} else 100, model_dir=args.out_dir)()
    elif args.training_method == "gairat":
        GairatTraining(args.dataset, args.model, model_dir=args.out_dir)()
    elif args.training_method == "trades":
        TradesTraining(args.dataset, args.model_dir, model_ir=args.out_dir)()
    elif args.training_method == "tradesawp":
        TradesawpTraining(args.dataset, args.model)()  # model dir not specifiable?
    elif args.training_method == "va":
        VaTraining(args.dataset, args.model, model_dir = args.out_dir)()
    elif args.training_method == "yopo":
        YopoTraining(args.dataset, args.model)()  # no model dir?


if __name__ == "__main__":
    main()
