# Neural-Machine-Translation-English-to-Hindi

Repo contains code to train different kinds of Neural Machine Translation Models. Translation is done from English to Hindi to see the models in action.

Following types of generation models were explored for this task:
- Basic Sequence-to-Sequence
- Sequence-to-Sequence + Basic Attention
- Sequence-to-Sequence + Global Attention
- Sequence-to-Sequence + Attention + Coverage

Following is the description of different files and folders in this project:

- `data/` folder contains the code used to train the models. The data is of small size as the main goal of the project was to explore different models to later scale them on larger datasets.

- `trained_models/` folder will contains the trained models once the training is done.

- `output/` folder contains the outputs that was given by all the different models.

- `vocab_generation.py` is used to assign IDs to tokens from the training set for english and hindi both.

- `train.py` file is the main file is the run to train different kinds of model.

- `model.py` file contains code for 5 of the models excluding coverage.

- `coverage_model.py` file contains code of the coverage model.

- `generate_output.py` file generates the output for test set and gives hindi translations for english sentences.

- `test_input.py` file can be used to get single sentence input translation from the trained model in real-time.
