## Running locally

Clone this repository and run the following commands:

1. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/active
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run sample prediction

If the trained model is not present locally, it will download from hugging-face from this
[model-repo](https://huggingface.co/deshrit/bee-count/tree/main) and run the inference.

```bash
./sample_solution.py <image file>
```

This will output as:
> Output file: predicted.jpg
> 
> Total bees: 48

## Running the test

```bash
python -m unittest
```