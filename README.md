# CNNs, RNNs, and Sequence Translation

This project covers various topics in deep learning, including image classification, sequence-to-sequence models, and autoencoders. The code is implemented in Python using the PyTorch library.

## 1. Age Prediction with Convolutional Neural Networks

In this problem, a Convolutional Neural Network (CNN) is implemented for age prediction using facial images. The dataset consists of grayscale face images and corresponding age labels. The VGG16 architecture is used as the base model, and the output layer is modified to produce a single age prediction.

1. **Custom Dataset Class:** Define a custom dataset class (`CustomDataset`) to load and preprocess the face images and age labels.

2. **Data Preprocessing:** Preprocess the data, including loading the VGG16 model with default weights, resizing images, and normalizing pixel values.

3. **Data Splitting:** Split the dataset into training, validation, and testing sets.

4. **Hyperparameter Tuning:** Use grid search to find the best hyperparameters (learning rate and batch size) for the model.

5. **Training:** Train the model using the selected hyperparameters and evaluate its performance on the test set.

### 2. Sequence Modeling with Recurrent Neural Networks

This problem focuses on sequence modeling using Recurrent Neural Networks (RNNs). Three different RNN models are implemented and compared: Vanilla RNN, Truncated RNN, and Padded RNN. The models are trained and evaluated on a sequence prediction task.

1. **Data Loading:** Load sequence data for training and testing.

2. **RNN Models:** Implement three RNN models: Vanilla RNN, Truncated RNN, and Padded RNN.

3. **Training:** Train each RNN model on the sequence data.

4. **Evaluation:** Evaluate the performance of each model on the test set.

5. **Analysis:** Compare the advantages and disadvantages of each RNN model.

### 3. Sequence-to-Sequence Translation and Autoencoder

This problem involves implementing a sequence-to-sequence model for translation between English and French. Additionally, an autoencoder model is trained to reconstruct input sequences. The models are trained on a dataset of English-French sentence pairs.

1. **Data Preparation:** Prepare language pairs and filter sentences based on length and prefixes.

2. **Encoder-Decoder Architecture:** Implement an Encoder-Decoder architecture for sequence-to-sequence translation.

3. **Autoencoder Model:** Train an autoencoder model for sequence reconstruction.

4. **Training:** Train both the translation and autoencoder models.

5. **Evaluation:** Evaluate the translation model's performance on sample sentences.

### Dependencies

- Python 3
- PyTorch
- NumPy
- Matplotlib

### Running the Code

Run each problem's code in a Jupyter notebook or a Python environment. Ensure that the dataset files are available, and modify file paths if necessary.
