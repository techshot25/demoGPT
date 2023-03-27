from argparse import ArgumentParser
import logging

import torch
import yaml

from gptmodel.train import train
from gptmodel.utils import sample_tokens


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(lineno)d @ %(filename)s - [%(levelname)s]: %(message)s",
    datefmt="%I:%M:%S%p",
)


if __name__ == "__main__":
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--save", action="store_true", help="Save model to file.")
    parser.add_argument(
        "-f", "--file-name", type=str, default="gpt.pt", help="Path to save model."
    )
    group.add_argument(
        "-c", "--cached", action="store_true", help="Load a cached model."
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=20,
        help="Max number of tokens to use for predictions.",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="config.yml",
        help="Path to configuration yaml file .",
    )

    args = parser.parse_args()

    with open(args.configuration, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using %s backend", device)
    model, vocab = train(config, device, cached=args.cached)
    if args.save:
        logging.info("Saving model to %s", parser.file_name)
        torch.save(model.state_dict(), parser.file_name)
    if args.cached:
        logging.info("Loading model from %s", parser.file_name)
        model = torch.load(parser.file_name)

    prompt = input("Enter some tokens to start: ")
    tokens = sample_tokens(
        model=model,
        vocab=vocab,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
    )
    print(tokens)
