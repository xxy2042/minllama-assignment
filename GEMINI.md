# Project Overview

This project is a minimalist implementation of the Llama2 model, developed as an assignment for Carnegie Mellon University's CS11-711 Advanced NLP course. The primary goal is to implement core components of the Llama2 architecture and use it for sentence classification tasks on the SST and CFIMDB datasets.

The project is written in Python and utilizes PyTorch for building the neural network. The model is pretrained on the TinyStories dataset.

# Building and Running

## Environment Setup

To set up the development environment and install dependencies, run the following command:

```bash
sh setup.sh
```

## Running the Model

The `run_llama.py` script is the main entry point for running the model. It has three main options: `generate`, `prompt`, and `finetune`.

### Text Generation

To generate text completions, use the `generate` option:

```bash
python run_llama.py --option generate
```

### Zero-Shot Prompting

To perform zero-shot sentiment analysis, use the `prompt` option with the appropriate dataset:

**SST:**
```bash
python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt
```

**CFIMDB:**
```bash
python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt
```

### Classification Finetuning

To finetune the model for classification, use the `finetune` option:

**SST:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt
```

**CFIMDB:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt
```

# Development Conventions

The core implementation of the Llama2 model is located in the following files:

*   `llama.py`: Contains the main Llama2 model architecture, including the Attention and LlamaLayer classes.
*   `classifier.py`: Contains the sentence classification head and finetuning pipeline.
*   `optimizer.py`: Contains the implementation of the AdamW optimizer.
*   `rope.py`: Contains the implementation of rotary positional embeddings.

The `structure.md` file provides a detailed description of the code structure and the parts that need to be implemented (marked with `#todo`).

The following files are provided and should not be modified:

*   `base_llama.py`
*   `tokenizer.py`
*   `config.py`
*   `utils.py`

Sanity checks are provided in `sanity_check.py` and `optimizer_test.py` to test the implementation.
