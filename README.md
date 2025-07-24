# 🏏 IPL Score Prediction using Deep Learning

This project aims to predict the **final score** of an IPL T20 match using deep learning techniques. The model is trained on historical IPL data and takes real-time match inputs to provide accurate score predictions.

---

## 🚀 Features

- Predicts **final match score** based on:
  - Batting & Bowling Teams
  - Venue
  - Current Score
  - Overs Completed
  - Wickets Fallen
  - Current Batsmen & Bowler
- Trained using a **Deep Neural Network** built with TensorFlow & Keras
- Clean and efficient **data preprocessing**
- Interactive prediction using `ipywidgets`
- 📊 Model performance visualization and accuracy analysis

---

## 🧪 Tech Stack

| Technology     | Purpose                              |
|----------------|--------------------------------------|
| Python         | Core Programming Language            |
| Pandas         | Data Handling                        |
| NumPy          | Numerical Computation                |
| TensorFlow/Keras| Deep Learning Model                 |
| Matplotlib     | Data Visualization                   |
| ipywidgets     | Interactive Prediction Widget        |

---

## 📂 Dataset

- **File:** `ipl_data.csv`
- Contains IPL match data including teams, players, runs, wickets, and venues.
- Basic data preprocessing and encoding done before feeding into the model.

---

## 🧠 Model Architecture

- **Input Layer:** One-hot encoded categorical + numerical features
- **Hidden Layers:** 3 Dense layers with ReLU activation
- **Output Layer:** Single neuron for score prediction (Regression)
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

---

## 📊 Results

- Achieved low **MSE loss** and high predictive accuracy on validation set
- Tested using various match situations
- Predictions are close to actual final scores in multiple test cases

---

## 🕹 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/TUFAQUE/IPL-Score-Prediction.git
   cd IPL-Score-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook IPL_Score_Prediction.ipynb
   ```

4. Scroll to the bottom and interact with the `ipywidgets` to try different match scenarios!

---

## 📌 Future Improvements

- Deploy as a web app using **Streamlit or Flask**
- Add real-time data scraping from live IPL matches
- Integrate player form & weather data
- Visual UI for non-technical users

---

## 👨‍💻 Author

**Tufaque A. Sayyed**  
AI & ML Engineer | Data Scientist |Ai Agent Builder

📧 Email: tufaquesayyed@gmail.com  
🔗 [GitHub](https://github.com/TUFAQUE) | [LinkedIn](https://www.linkedin.com/in/tufaque-sayyed-843596364/)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it with proper attribution.

> See the [LICENSE](LICENSE) file for full license text.

---

## 📁 Folder Structure

```
├── IPL_Score_Prediction.ipynb     # Jupyter Notebook with code and predictions
├── ipl_data.csv                   # Dataset file
├── requirements.txt               # Python dependencies (to be added)
└── README.md                      # Project documentation
```

---

## 🙌 Support

If you like this project, don’t forget to ⭐ star the repo and follow for more!

```bash
git clone https://github.com/TUFAQUE/IPL-Score-Prediction.git
```

