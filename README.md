# Rice Leaf Disease Detection using InceptionV3

This project uses a deep learning model (InceptionV3) to detect diseases in rice leaf images. The model is trained using Keras and TensorFlow, and the web interface is built with Streamlit.

## Features

- Detects: **Brown Spot**, **Healthy**, and **Leaf Blast** rice leaves.
- Uses transfer learning with InceptionV3.
- Provides a simple web interface for image upload and prediction.

## Project Structure

```
.
├── Advanced_inceptioV3_sgd.py   # Model training and evaluation script
├── app.py                       # Streamlit web app for prediction
├── Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras  # Trained model file
├── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow pillow numpy scikit-learn matplotlib seaborn
```

### 3. Prepare the dataset

Organize your dataset in the following structure:

```
split_dataset/
    train/
        BrownSpot/
        Healthy/
        LeafBlast/
    validation/
        BrownSpot/
        Healthy/
        LeafBlast/
    test/
        BrownSpot/
        Healthy/
        LeafBlast/
```

Update the paths in `Advanced_inceptioV3_sgd.py` if your dataset is in a different location.

### 4. Train the Model

If you don't have the `.keras` model file, run:

```bash
python Advanced_inceptioV3_sgd.py
```

This will train the model and save it as `Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras`.

### 5. Run the Web App

```bash
streamlit run app.py
```

Upload a rice leaf image and get instant predictions!

## Notes

- Make sure the model file (`Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras`) is in the same directory as `app.py`.
- The class labels in `app.py` must match the folder names used during training.

## License

This project is for academic and research purposes.
