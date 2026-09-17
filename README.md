# AI MIDI Music Generator

This repository contains a Deep Learning project that generates original piano music using Long Short-Term Memory (LSTM) neural networks. By feeding the model a dataset of MIDI files, it learns the underlying patterns, structures, and chord progressions of the music, allowing it to predict and compose new, unique sequences.

## 🧠 How it Works

The project is structured into three main phases:

1. **Data Preprocessing:** Uses `music21` to parse `.mid` files, extracting all notes and chords. These musical elements are mapped to integer categories. The data is then structured into sequences of 100 notes, which serve as the input to predict the 101st note.
2. **Model Architecture:** A multi-layer Deep LSTM network built with Keras. It utilizes 3 LSTM layers (512 units each) intermixed with Dropout layers (0.3) to prevent overfitting. The model outputs a softmax probability distribution over the vocabulary of unique notes.
3. **Music Generation:** The trained model is seeded with a random sequence of 100 notes. It predicts the next note, appends it to the sequence, and shifts the window forward. This loop repeats to generate a 200-note composition, which is then translated back into a playable MIDI file.

## 🛠️ Dependencies

To run this project, you will need Python 3.x and the following libraries installed:

* `tensorflow` / `keras`
* `music21`
* `numpy`

You can install them via pip:
```bash
pip install tensorflow keras music21 numpy
