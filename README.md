### 🐶 Dog Breed Classification Using Transfer Learning and TensorFlow 2.0
This project aims to classify different dog breeds using `transfer learning` and `TensorFlow 2.0.` We leverage a dataset of over `10,000` labeled images of `120` different dog breeds from the Kaggle dog breed identification competition.
##

### Workflow Overview:

#### 1. Data Preparation.

#### Download the dataset from Kaggle.
> https://www.kaggle.com/c/dog-breed-identification/data

* Store and import the data into the data folder.
* Preprocess the data by creating the train, validation, and test sets.

#### 2. Model Selection and Training

Use pre-trained deep learning models from `TensorFlow Hub` (tf.keras.applications) to build our model.
Train the model using tools such as `TensorBoard` and `EarlyStopping`.

##### 3. Model Evaluation

Make predictions on test data and compare results with `ground truth labels` to evaluate performance.

##### 4. Model Improvement

Start by experimenting with a small dataset `(1,000 images)`, and increase dataset size once the model is performing as expected.

##### 5. Model Saving and Sharing

Save the trained model and reload it for future predictions.
The exported or saved models are stored in the models/ folder, in both `.h5` and `Kera`s format for easy reloading and use in other projects.

#### Additional Project Files:

##### Log Folder:

During training, logs are saved for both the training and validation sets.
The logs can be found in the logs/ directory, which includes files for tracking metrics such as accuracy and loss using TensorBoard.

##### Models Folder:

The trained models are saved in the models/ folder, stored in both .h5 format and the Keras model format.
These can be reloaded for making future predictions without retraining the model.

#### Tools and Libraries Used:

* TensorFlow 2.x for data preprocessing and building deep learning models.
* TensorFlow Hub for utilizing pretrained models.
* Google Colab as the recommended IDE for easy cloud-based experimentation.

#### Dataset:
The dataset is sourced from the Kaggle Dog Breed Identification.

##### How to Run:
* Download the dataset from Kaggle.
* Open the Jupyter Notebook or run the code on Google Colab.
* Follow the steps to preprocess data, train the model, evaluate, and save results.
* Use the logs stored in the logs/ folder to track model performance.
* Reload and use the trained model from the models/ folder for future predictions.
##
##

#### Colab Troubleshooting & Common Issues:
When running the notebook in Google Colab, you might face a few common issues. Here are some possible troubleshooting tips:

##### Runtime Disconnects:
Colab environments may sometimes disconnect, especially during long training sessions. You can avoid losing progress by frequently saving model checkpoints and logs. Use the models/ folder to save trained models periodically.

> Solution: Set up automatic saving of model checkpoints using Keras ModelCheckpoint callback.

##### Memory Limit Exceeded:
While working with large datasets like the dog breed classification set, you might hit Colab's memory limit.

> Solution: Use a smaller batch size or a reduced number of images for training. Start with 1,000 images and gradually increase as the model performs well.

##### Kaggle API Not Installed:
You may get an error about the Kaggle API not being installed when trying to download data directly from Kaggle.

>Solution: Install the Kaggle API in Colab using !pip install kaggle. Also, ensure that your Kaggle API key is correctly uploaded to the Colab environment.

##### File Access Errors:
Make sure the logs/ and models/ folders exist before trying to write files into them.

>Solution: If you face issues writing files, ensure the directories are created using:

~~~~
python
Copy code
import os
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('models'):
    os.makedirs('models')
~~~~

##### Slow Training Speeds:
If training is slow, try using a GPU runtime in Colab.

> Solution: Go to Runtime > Change runtime type > Hardware accelerator > GPU.

#### License
This project is open-source under the `MIT License`.
