---
name: transformers
description: Agent-invokable Transformers commands. Pass `--format json` at the top level (e.g. `transformers --format json classify ...`) to receive the structured output documented in each capability's `outputs` schema.
---

# Transformers CLI

For one-off inference, training, quantization, or export, invoke the
`transformers` command directly rather than writing Python. Run
`transformers --help` for the full command list; run
`transformers <command> --help` for flags per command.

## Invocation rules

**All inputs are named flags, never positional.** Wrong invocations like
``transformers classify "my text"`` or ``transformers ner "sentence"`` will
fail with ``Got unexpected extra argument``. The text / image / audio / file
argument is always a flag: ``--text``, ``--image``, ``--audio``, ``--file``.

**Always invoke as `transformers <cmd> ...`.** Do not use
``python -m transformers ...`` patterns — the console script is what the
``transformers`` package installs.

**Use `transformers --format json` for machine-readable output**:
``transformers --format json classify --text "..."``.

## Example invocations (copy these shapes)

Text (classify, ner, token-classify, summarize, translate, fill-mask):
```
transformers classify --text "I loved this movie"
transformers classify --text "..." --model distilbert/distilbert-base-uncased-finetuned-sst-2-english
transformers ner --text "Apple CEO Tim Cook visited Paris." --model dslim/bert-base-NER
transformers summarize --file article.txt --model facebook/bart-large-cnn
transformers translate --text "The weather is nice" --model Helsinki-NLP/opus-mt-en-de
```

Question answering (takes `--question` and `--context`):
```
transformers qa --question "Who invented it?" --context "Graham Bell invented the telephone in 1876."
```

Image (caption, image-classify, detect, segment, depth, vqa, ocr):
```
transformers caption --image photo.jpg --model llava-hf/llava-interleave-qwen-0.5b-hf
transformers image-classify --image photo.jpg
transformers vqa --image photo.jpg --question "What color is the car?"
```

Audio (transcribe, audio-classify, speak):
```
transformers transcribe --audio clip.wav --model openai/whisper-tiny
transformers audio-classify --audio clip.wav
transformers speak --text "Hello" --output hello.wav
```

Tokenize / inspect / embed:
```
transformers tokenize --model HuggingFaceTB/SmolLM2-360M-Instruct --text "tokenize me"
transformers inspect meta-llama/Llama-3.2-1B-Instruct
transformers embed --text "some sentence" --model BAAI/bge-small-en-v1.5
```

Generate (text completion):
```
transformers generate --prompt "Once upon a time" --model HuggingFaceTB/SmolLM2-360M-Instruct
```

## Available commands

- `transformers classify` — Classify text into categories
- `transformers ner` — Extract named entities from text (NER)
- `transformers token-classify` — Tag tokens with labels (POS tagging, chunking, etc.)
- `transformers qa` — Answer a question given a context paragraph (extractive QA)
- `transformers table-qa` — Answer a question about tabular data (CSV)
- `transformers summarize` — Summarize text
- `transformers translate` — Translate text between languages
- `transformers fill-mask` — Predict the masked token in a sentence
- `transformers image-classify` — Classify an image
- `transformers detect` — Detect objects in an image
- `transformers segment` — Segment an image
- `transformers depth` — Estimate a depth map from an image
- `transformers keypoints` — Match keypoints between two images
- `transformers video-classify` — Classify a video
- `transformers transcribe` — Transcribe speech from an audio file
- `transformers audio-classify` — Classify an audio file into categories
- `transformers speak` — Synthesize speech from text and save to a WAV file
- `transformers audio-generate` — Generate audio (e.g. music) from a text description and save to a WAV file
- `transformers vqa` — Visual question answering using ``AutoModelForImageTextToText``
- `transformers document-qa` — Extractive document question answering using
- `transformers caption` — Generate a caption for an image using ``AutoModelForImageTextToText``
- `transformers ocr` — Extract text from an image using ``AutoModelForImageTextToText``
- `transformers multimodal-chat` — Single-turn conversation with a model that accepts mixed inputs
- `transformers generate` — Generate text from a prompt with full control over decoding
- `transformers detect-watermark` — Detect whether text contains a watermark
- `transformers embed` — Compute embeddings for text or images
- `transformers tokenize` — Tokenize text and display the resulting tokens
- `transformers inspect` — Inspect a model's configuration without downloading weights
- `transformers inspect-forward` — Examine attention weights and hidden states from a forward pass
- `transformers benchmark-quantization` — Compare quality and performance across quantization methods
- `transformers train` — Fine-tune or pretrain a model on a dataset
- `transformers quantize` — Quantize a model and save it
- `transformers export` — Export a model to a deployment-friendly format

## When to use what

- **Atomic task** (single inference / training / export): use the CLI.
- **Composed workflow** (chain models, custom logic): write Python.
  The CLI commands' source in `transformers.cli.agentic.*` is the
  canonical template — each file loads a model with `AutoModel*` +
  `AutoProcessor`/`AutoTokenizer`, runs a forward pass, and
  post-processes. Copy that pattern rather than reaching for
  `pipeline(...)`.
