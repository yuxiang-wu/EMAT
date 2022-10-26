import argparse
import logging
import math
import os
import random
import tempfile
from functools import partial

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    T5Tokenizer,
    T5Config,
)
from transformers.utils.versions import require_version

from emat.t5_cat import CaseAugmentedT5, Seq2SeqLMOutputWithCustomLosses, replicate_first_half
from emat.utils import write_jsonl

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Initialise wandb
try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False

DATA_PATHS = {
    "nq": {
        "train": "nq/NQ-open.train-train.jsonl",
        "validation": "nq/NQ-open.train-dev.jsonl",
        "test": "nq/NQ-open.test.jsonl"
    },
    "tqa": {
        "train": "tqa/triviaqa.train-train.jsonl",
        "validation": "tqa/triviaqa.train-dev.jsonl",
        "test": "tqa/triviaqa.test.jsonl"
    },
    "paq-l1": {
        "train": "paq/PAQ_L1/PAQ_L1.filtered.jsonl",
    },
    "paq": {
        "train": "paq/PAQ/PAQ.filtered.jsonl",
    },
}


def load_train_validation_data(args) -> DatasetDict:
    assert args.train_data is not None
    if args.validation_data is None:
        args.validation_data = args.train_data
    data_files = {"train": [], "validation": []}

    for split, data_str in [("train", args.train_data), ("validation", args.validation_data)]:
        for name in data_str.split(","):
            if name in DATA_PATHS and DATA_PATHS[name].get(split, None) is not None:
                data_files[split].append(os.path.join(args.data_dir, DATA_PATHS[name][split]))
            elif os.path.exists(name) and os.path.isfile(name):  # is a file
                assert name.endswith(".json") or name.endswith(".jsonl"), "Can only load json or jsonl files."
                data_files[split].append(name)
            else:
                raise ValueError(f"Failed to load data {name} split {split}")

    return load_dataset("json", data_files=data_files)


def load_test_data(args) -> DatasetDict:
    assert args.test_data is not None
    data_files = {"test": []}

    split, data_str = "test", args.test_data
    for name in data_str.split(","):
        if name in DATA_PATHS and DATA_PATHS[name].get(split, None) is not None:
            data_files[split].append(os.path.join(args.data_dir, DATA_PATHS[name][split]))
        elif os.path.exists(name) and os.path.isfile(name):  # is a file
            assert name.endswith(".json") or name.endswith(".jsonl"), "Can only load json or jsonl files."
            data_files[split].append(name)
        else:
            raise ValueError(f"Failed to load data {name} split {split}")

    return load_dataset("json", data_files=data_files)


def save_model(model, save_dir, accelerator=None, tokenizer=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if accelerator is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
    else:
        model.save_pretrained(save_dir)

    if tokenizer is not None:
        if accelerator is None:
            tokenizer.save_pretrained(save_dir)
        elif accelerator.is_local_main_process:
            tokenizer.save_pretrained(save_dir, save_function=accelerator.save)


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a sequence-to-sequence task")
    parser.add_argument("--project_name", type=str, default="CAT", help="Project name.")
    parser.add_argument("--exp_name", type=str, default=None, required=True, help="Experiment name.")

    parser.add_argument("--train_data", type=str, default=None, required=True, help="Training dataset name or path.")
    parser.add_argument("--validation_data", type=str, default=None, help="Dev dataset name or path.")
    parser.add_argument("--test_data", type=str, default=None, help="Test dataset name or path.")
    parser.add_argument("--data_dir", type=str, default="data", help="path to the data directory.")

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
             "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
             "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
             "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="Whether to pad all samples to model maximum sentence "
             "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
             "efficient on GPU but very bad for TPU.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--early_stop_patience", type=int, default=1000000,
                        help="Early stop if the performance does not improve for this number of epochs .")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to train the model on the train set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from the latest checkpoint.")

    # (jimmycode): new CAT-related arguments
    parser.add_argument("--prefix_length", type=int, default=4, help="Length of the prefix.")
    # MLE loss
    parser.add_argument("--mle_weight", type=float, default=1.0, help="Weight for the MLE loss.")
    # Autoencoder loss
    parser.add_argument("--use_autoencoder", action="store_true", help="Train the model with auto-encoding objective.")
    parser.add_argument("--autoencoder_weight", type=float, default=1.0, help="Weight for the auto-encoding objective.")
    # Key matching loss
    parser.add_argument("--use_cat_key", action="store_true", help="Train the model with key matching loss.")
    parser.add_argument("--cat_key_layer", type=int, default=None, help="The layer that computes the key embedding.")
    parser.add_argument("--cat_d_key", type=int, default=None, help="The dimension of key embeddings.")
    parser.add_argument("--cat_key_margin", type=float, default=0.1, help="Margin for the key matching ranking loss.")
    parser.add_argument("--cat_key_weight", type=float, default=1.0, help="Weight for the key matching objective.")
    # Value swapping loss
    parser.add_argument("--use_cat_value", action="store_true",
                        help="Train the model by swapping value embeddings with case memory.")
    parser.add_argument("--cat_value_layer", type=int, default=None,
                        help="The layer that it conducts value embedding swapping.")
    parser.add_argument("--cat_value_weight", type=float, default=1.0, help="Weight for the value swap objective.")

    parser.add_argument("--validation_metric", type=str, default="mle_em",  # choices=["mle_em", "ae_em", "value_em"],
                        help="Metric for validation.")

    args = parser.parse_args()

    # Sanity checks
    if args.train_data is None and args.test_data is None:
        raise ValueError("Need either a training/test dataset name or file.")
    if args.use_cat_value and not args.use_autoencoder:
        raise ValueError("use_autoencoder needs to be True if use_cat_value is True.")

    return args


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process and _has_wandb:
        wandb.init(project=args.project_name, name=args.exp_name, dir=args.output_dir, config=vars(args))

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.resume_training and args.output_dir is not None:
        args.model_name_or_path = os.path.join(args.output_dir, "latest_ckpt")

    if args.config_name:
        config = T5Config.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = T5Config.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # (jimmycode): set CAT-related config
    config.prefix_length = args.prefix_length
    config.key_layer = args.cat_key_layer if args.cat_key_layer is not None else 1
    config.d_key = args.cat_d_key
    config.value_layer = args.cat_value_layer if args.cat_value_layer is not None else config.num_layers - 1
    # use the last layer if cat_value_layer is not specified.

    if args.tokenizer_name:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = CaseAugmentedT5.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = CaseAugmentedT5.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else True

    # Create data collator
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
    # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    pad_to_multiple_of = 8 if accelerator.use_fp16 else None

    def preprocess_function(examples, use_autoencoder=args.use_autoencoder, use_cat_value=args.use_cat_value):
        questions = [ex["question"] for ex in examples]
        answers = [ex["answer"][0] for ex in examples]

        def _verbalise(q, a):
            q = q.strip("?")
            return f'{q}? context: "{a}" is the answer to question {q}'

        def _process_labels(labels):
            input_ids = labels["input_ids"]
            bsz, label_length = input_ids.size()

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if args.ignore_pad_token_for_loss:
                input_ids[input_ids == tokenizer.pad_token_id] = label_pad_token_id

            if pad_to_multiple_of is not None:
                max_label_length = (
                        (label_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
                )
                remainder = max_label_length - label_length
                if remainder > 0:
                    pad_ids = torch.full(
                        (bsz, remainder),
                        fill_value=label_pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device
                    )
                    input_ids = torch.cat([input_ids, pad_ids], dim=1)

            return input_ids

        # Normal inputs and outputs
        inputs = [prefix + q for q in questions]
        targets = answers

        # Autoencoder input and outputs
        if use_autoencoder:
            ae_inputs = [prefix + _verbalise(q, a) for q, a in zip(questions, answers)]
            ae_targets = answers
            inputs.extend(ae_inputs)
            targets.extend(ae_targets)

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True,
                                 return_tensors="pt")
        with tokenizer.as_target_tokenizer():  # setup the tokenizer for targets
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,
                               return_tensors="pt")
        model_inputs["labels"] = _process_labels(labels)

        if use_cat_value:
            for k, v in model_inputs.items():
                model_inputs[k] = replicate_first_half(v)

        return model_inputs

    # Evaluation metric: load the custom exact-match metric
    metric = load_metric("./emat/evaluation/exact_match.py")
    ae_metric = load_metric("./emat/evaluation/exact_match.py") if args.use_autoencoder else None
    value_metric = load_metric("./emat/evaluation/exact_match.py") if args.use_cat_value else None

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    if args.do_train:
        # Load the training/validation datasets
        raw_datasets = load_train_validation_data(args)
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=preprocess_function, batch_size=args.per_device_train_batch_size,
            num_workers=args.preprocessing_num_workers
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=preprocess_function,
                                     batch_size=args.per_device_eval_batch_size,
                                     num_workers=args.preprocessing_num_workers)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        # (jimmycode): Hit@1 metric
        hit1_metric = load_metric("./emat/evaluation/accuracy.py") if args.use_cat_key else None

        best_em, patience = 0., args.early_stop_patience
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                if epoch == 0 and step == 0:
                    print(batch)
                    print(f"input_ids shape: {batch['input_ids'].shape}")
                    print(f"labels shape: {batch['labels'].shape}")

                outputs: Seq2SeqLMOutputWithCustomLosses = model(
                    **batch,
                    use_autoencoder=args.use_autoencoder,
                    use_cat_key=args.use_cat_key,
                    cat_key_margin=args.cat_key_margin,
                    use_cat_value=args.use_cat_value,
                )

                # Weight the losses
                loss_dict = outputs.loss_dict
                loss = loss_dict["mle_loss"] * args.mle_weight

                if args.use_autoencoder:
                    loss += args.autoencoder_weight * loss_dict["ae_loss"]
                    if args.use_cat_value:
                        loss += loss_dict["value_loss"] * args.cat_value_weight

                if args.use_cat_key:
                    loss += loss_dict["key_match_loss"] * args.cat_key_weight
                    hit1_metric.add_batch(
                        predictions=torch.argmax(outputs.key_match_matrix, dim=-1).tolist(),
                        references=range(outputs.key_match_matrix.shape[1])
                    )  # compute the Hit@1 accuracy for key matching

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    if accelerator.is_local_main_process and _has_wandb:
                        wandb.log({"loss": loss * args.gradient_accumulation_steps, "step": completed_steps})
                        for k, v in loss_dict.items():
                            wandb.log({k: v, "step": completed_steps})

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()

            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": args.val_max_target_length if args is not None else config.max_length,
                "num_beams": args.num_beams,
            }
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_autoencoder=args.use_autoencoder,
                        use_cat_key=args.use_cat_key,
                        use_cat_value=args.use_cat_value,
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    labels = batch["labels"]
                    if not args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                  pad_index=tokenizer.pad_token_id)

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    if args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    if args.use_autoencoder:
                        if not args.use_cat_value:
                            bsz = len(decoded_preds) // 2
                            metric.add_batch(predictions=decoded_preds[:bsz], references=decoded_labels[:bsz])
                            ae_metric.add_batch(predictions=decoded_preds[bsz:], references=decoded_labels[bsz:])
                        else:
                            bsz = len(decoded_preds) // 3
                            metric.add_batch(predictions=decoded_preds[:bsz], references=decoded_labels[:bsz])
                            ae_metric.add_batch(predictions=decoded_preds[bsz:2 * bsz],
                                                references=decoded_labels[bsz:2 * bsz])
                            value_metric.add_batch(predictions=decoded_preds[2 * bsz:],
                                                   references=decoded_labels[2 * bsz:])
                    else:
                        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            mle_em = metric.compute()["em"] * 100  # EM score is not in percentage points
            logger.info(f"epoch {epoch} eval - EM: {mle_em:.2f}")
            if accelerator.is_local_main_process and _has_wandb:
                wandb.log({"mle_em_dev": mle_em, "epoch": epoch})

            ae_em = None
            if ae_metric is not None:
                ae_em = ae_metric.compute()["em"] * 100
                logger.info(f"epoch {epoch} eval - Autoencoder EM: {ae_em:.2f}")
                if accelerator.is_local_main_process and _has_wandb:
                    wandb.log({"ae_em_dev": ae_em, "epoch": epoch})

            value_em = None
            if value_metric is not None:
                value_em = value_metric.compute()["em"] * 100
                logger.info(f"epoch {epoch} eval - Value Swap EM: {value_em:.2f}")
                if accelerator.is_local_main_process and _has_wandb:
                    wandb.log({"value_em_dev": value_em, "epoch": epoch})

            hit1_acc = None
            if hit1_metric is not None:
                hit1_acc = hit1_metric.compute()["accuracy"] * 100
                logger.info(f"epoch {epoch} eval - Key Matching Hit@1 Accuracy: {hit1_acc:.2f}")
                if accelerator.is_local_main_process and _has_wandb:
                    wandb.log({"hit@1_train": hit1_acc, "epoch": epoch})

            if args.output_dir is not None:
                save_model(model, os.path.join(args.output_dir, "latest_ckpt"), accelerator, tokenizer=tokenizer)

                all_metrics = {"mle_em": mle_em, "ae_em": ae_em, "value_em": value_em}
                best_metric = sum(
                    [all_metrics[m] for m in args.validation_metric.split(",") if all_metrics[m] is not None]
                )
                # if best_metric is None:
                #     best_metric = mle_em

                if best_metric > best_em:
                    best_em = best_metric
                    save_model(model, os.path.join(args.output_dir, "best_ckpt"), accelerator, tokenizer=tokenizer)
                    patience = args.early_stop_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        break

    if args.do_predict:
        if 'accelerator' in locals() and not accelerator.is_local_main_process:
            return
        device = accelerator.device if 'accelerator' in locals() else "cuda"

        best_ckpt_dir = os.path.join(args.output_dir,
                                     "best_ckpt") if args.output_dir is not None else args.model_name_or_path
        logger.info(f"Loading best checkpoint from {best_ckpt_dir}")

        best_config = AutoConfig.from_pretrained(best_ckpt_dir)
        best_model = AutoModelForSeq2SeqLM.from_pretrained(
            best_ckpt_dir,
            from_tf=bool(".ckpt" in best_ckpt_dir),
            config=best_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(best_ckpt_dir, use_fast=not args.use_slow_tokenizer)

        best_model.to(device)
        best_model.eval()

        # Load the test dataset
        raw_datasets = load_test_data(args)

        test_dataset = raw_datasets["test"]
        test_dataloader = DataLoader(test_dataset,
                                     collate_fn=partial(
                                         preprocess_function, use_autoencoder=False, use_cat_value=False
                                     ),
                                     batch_size=args.per_device_eval_batch_size,
                                     num_workers=args.preprocessing_num_workers)

        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
            # "use_autoencoder": False,
        }

        all_questions, all_answers, all_preds = [], [], []
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_tokens = best_model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            start_idx = len(all_questions)
            questions, answers = [], []
            for i in range(len(decoded_preds)):
                example = test_dataset[start_idx + i]
                question = example["question"]
                questions.append(question)
                answer = example["answer"]
                answers.append(answer)

            metric.add_batch(predictions=decoded_preds, references=answers)
            all_questions.extend(questions)
            all_answers.extend(answers)
            all_preds.extend(decoded_preds)

        assert len(all_questions) == len(test_dataset), "Prediction size does not match data size"

        test_metric = metric.compute()
        test_em = test_metric["em"] * 100  # EM score is not in percentage points
        logger.info(f"test - EM: {test_em:.2f}")
        if _has_wandb:
            wandb.log({"em_test": test_em})

        if args.output_dir is None:
            _, pred_file = tempfile.mkstemp(prefix=f"{args.project_name}_")
        else:
            pred_file = os.path.join(args.output_dir, "preds.jsonl")

        predictions = [{"question": q, "answer": a, "prediction": p} for q, a, p in
                       zip(all_questions, all_answers, all_preds)]
        logger.info(f"Write predictions into {pred_file}")
        write_jsonl(predictions, pred_file)


if __name__ == "__main__":
    main()
