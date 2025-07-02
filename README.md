# 🐶 Dog Breed Classification - Transfer Learning Project

Welcome to my **Dog Breed Classification** project built with TensorFlow and transfer learning. This model takes in a dog photo and predicts its breed among 120 categories.

> 🧪 A great learning exercise to apply transfer learning, image preprocessing, and evaluation metrics on real-world image data.

---

## 📌 About the Project

- 📚 **Dataset**: [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)
- 🧠 **Model**: Transfer Learning using TensorFlow Hub (`KerasLayer`)
- 🎯 **Objective**: Predict the breed of a dog given its photo
- 📈 **Evaluation**: Classification report (precision, recall, F1-score)

---

## 🧰 Technologies Used

| Tool         | Purpose                        |
|--------------|--------------------------------|
| Python       | Programming language           |
| TensorFlow   | Deep learning framework        |
| Keras        | High-level neural network API  |
| scikit-learn | Evaluation metrics             |
| Matplotlib   | Visualization                  |
| NumPy        | Numerical operations           |

---

## 📁 Folder Structure

```

Dog-Breed-Classification/
│
├── config.py               # Project configuration
├── main.py                 # Loads model & predicts breed from new image
├── src/
│   ├── data\_pipeline.py    # Data preprocessing & batch generation
│   ├── model.py            # Build the CNN model
│   ├── train.py            # Train the model
│   ├── evaluate.py         # Evaluate trained model
│   └── predict.py          # Predict from custom image
│
├── scripts/
│   └── prepare\_data.py     # Load CSV and split into train/val
│
├── Data/                   # Processed training and test folders
├── saved\_model/            # Trained model (.h5)
├── requirements.txt        # Required libraries
└── README.md               # You are here

````

---

## 🖼️ Sample Prediction Output

> A fun prediction using a custom photo of me and my dog 🐾

⬇️ _Insert your saved figure here:_  
![Prediction Output](assets/loka_prediction.png) <!-- Rename or move accordingly -->

---

## 🚀 Run It Yourself

### 1. Clone the repo

```bash
git clone https://github.com/muhammederbay10/Dog-Breed-Classification.git
cd Dog-Breed-Classification
````

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

Make sure your data is placed in the following format:

```
Data/
└── processed/
    ├── train/
    ├── test/
    └── labels.csv
```

Then run:

```bash
python scripts/prepare_data.py
```

### 4. Train the model

```bash
python src/train.py
```

### 5. Evaluate the model

```bash
python src/evaluate.py
```

### 6. Predict custom image

```bash
python main.py
```

---

## 📊 Evaluation

* Accuracy: Varies by run (\~85% on validation)
* Example prediction: `golden_retriever`

---


## ⭐ Give it a Star!

If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!

```
