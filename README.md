🧠 IPL Score Prediction using Deep Learning
Last Updated: 24 July, 2025

In today’s fast-paced cricket environment, every decision—especially around scoring—can define the match's fate. Using deep learning, this project predicts IPL match scores in real-time, offering reliable insights to fans, broadcasters, and analysts.

🚀 Why Deep Learning for Score Prediction?
Unlike traditional machine learning models, deep learning identifies intricate patterns from large datasets and can adaptively learn the game dynamics—batting styles, bowling performance, match venues, etc. It allows us to predict scores with better accuracy and handle nonlinear dependencies.

📦 Project Structure
bash
Copy
Edit
├── IPL_Score_Prediction.ipynb       # Jupyter notebook containing the full pipeline
├── ipl_data.csv                     # Dataset used for training
├── README.md                        # Project overview
└── final_model.h5 (optional)        # Saved trained DL model
📚 Step-by-Step Implementation
1. 🛠 Installing Libraries
The following libraries were used:

python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
import tensorflow as tf
2. 📂 Loading the Dataset
The dataset (2008–2017) includes:

Teams

Venue

Batsmen & Bowlers

Overs, Runs, Wickets, Striker Info

python
Copy
Edit
ipl = pd.read_csv('ipl_data.csv')
3. 📊 Exploratory Data Analysis (EDA)
Visualized number of matches at each venue

Top 10 Batsmen by runs

Top 10 Bowlers by wickets

Uses matplotlib and seaborn for insightful plots.

4. 🔣 Label Encoding
Categorical features like teams, venues, players are encoded for neural networks.

python
Copy
Edit
label_encoders = {}
for col in ['bat_team', 'bowl_team', 'venue', "batsman", "bowler"]:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    label_encoders[col] = le
5. 🎯 Feature Selection & Correlation Analysis
Redundant or highly correlated features like non-striker, runs_last_5, wickets_last_5 were removed using heatmaps.

6. 🔀 Train-Test Split
Used train_test_split() with:

70% training

30% testing

random_state=42 for reproducibility

7. 📏 Feature Scaling
MinMax scaling ensures all input features are normalized before training:

python
Copy
Edit
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
8. 🧠 Neural Network Model
A Deep Learning Regressor built using TensorFlow + Keras:

python
Copy
Edit
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

huber_loss = tf.keras.losses.Huber(delta=1.0)
model.compile(optimizer='adam', loss=huber_loss)
9. 🏋️‍♂️ Model Training
Trained for 10 epochs using batch_size=64, and plotted loss vs validation loss:

python
Copy
Edit
model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test))
🔍 Evaluation Metrics
Predicted on the test set and evaluated using:

python
Copy
Edit
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
# MAE ≈ 14.41
🧪 Final Results
📈 Mean Absolute Error: ~14.41
🧠 Model Type: Deep Neural Network
🏏 Predicted Scores: Close to real-world IPL scores, tested on CSK vs RCB at Chinnaswamy Stadium
✅ Accuracy: Very good on unseen data

🧩 Interactive Score Prediction Widget
An interactive widget was built using ipywidgets in Jupyter to allow users to simulate match conditions and get live score predictions:

Dropdowns for Venue, Teams, Batsman, Bowler

Inputs for Overs, Runs, Wickets

Live Output of Predicted Total Score

python
Copy
Edit
display(venue, batting_team, bowling_team, striker, bowler,
        runs, wickets, overs,
        striker_ind,
        predict_button, output)
🛠 Tools Used
Tool	Purpose
Pandas & NumPy	Data Manipulation
Matplotlib & Seaborn	Data Visualization
Scikit-learn	Preprocessing & Evaluation
Keras / TensorFlow	Deep Learning Modeling
Ipywidgets	UI Widgets in Jupyter

💡 Future Scope
Deploy this as a web app using Streamlit or Flask

Add live data scraping from APIs for real-time prediction

Incorporate more features like player form, pitch type, match day weather

👨‍💻 Author
Tufaque A. Sayyed
AI & ML Engineer | Data Scientist | Ai Agent Builder
📧 Email: tufaquesayyed@gmail.com
🔗 GitHub | LinkedIn

🌟 Star This Repo
If this helped you, give it a ⭐ on GitHub and share with your fellow data science enthusiasts.

