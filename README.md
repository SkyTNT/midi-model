# Midi-Model

## Midi event transformer for music generation

![](./banner.png)

## Updates

- v1.2 : Optimise the tokenizer and dataset. The dataset was filtered by MIDITokenizer.check_quality. Using the higher quality dataset to train the model, the performance of the model is significantly improved.

## Demo

- [online: huggingface](https://huggingface.co/spaces/skytnt/midi-composer)

- [online: colab](https://colab.research.google.com/github/SkyTNT/midi-model/blob/main/demo.ipynb)

- [download windows app](https://github.com/SkyTNT/midi-model/releases)

## Pretrained model

[huggingface](https://huggingface.co/skytnt/midi-model/blob/main/model.ckpt)

## Dataset

[projectlosangeles/Los-Angeles-MIDI-Dataset](https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset)

## Requirements

- install [pytorch](https://pytorch.org/)(recommend pytorch>=2.0)
- install [fluidsynth](https://www.fluidsynth.org/)>=2.0.0
- `pip install -r requirements.txt`

## Run app

`python app.py`

## Train 

`python train.py`
 
