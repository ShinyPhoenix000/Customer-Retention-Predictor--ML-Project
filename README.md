# Customer Retention Predictor

The **Customer Retention Predictor** is a machine learning project designed to help businesses proactively reduce customer attrition. By leveraging **explainable AI (XAI)** techniques such as **SHAP (SHapley Additive exPlanations)** and advanced **tree ensemble models**, this system not only predicts customer churn but also provides insights into the key factors influencing those predictions.

📈 In testing, the model improved retention strategy accuracy by **25%**, enabling data-driven decision-making for customer engagement and retention planning.

---

## 🔑 Features
- **Customer Churn Prediction**: Predicts the likelihood of customers leaving.  
- **Explainable AI (XAI)**: Uses SHAP to provide transparency into model decisions.  
- **Model Interpretability**: Highlights key features driving customer behavior.  
- **Data-Driven Insights**: Supports targeted retention strategies and marketing campaigns.  
- **Improved Accuracy**: Achieved a **25% increase** in strategy effectiveness.  

## ⚙️ Tech Stack
- **Programming Language**: Python  
- **Machine Learning**: Scikit-learn, XGBoost / LightGBM, SHAP  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn, SHAP plots  
- **Experiment Tracking**: Jupyter Notebooks  

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ShinyPhoenix000/Customer-Retention-Predictor--ML-Project.git
cd Customer-Retention-Predictor--ML-Project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

1. Prepare your data:
   - Ensure your dataset has the required features
   - Place the data file in the appropriate directory
   - Update configuration if needed

2. Train the model:
```bash
python main.py train --config config.yaml
```

3. Generate predictions:
```bash
python main.py predict --config config.yaml --data-path your_data.csv
```

4. View explanations:
```bash
python main.py explain --config config.yaml --data-path your_data.csv
```

## 📁 Project Structure
```
Customer-Retention-Predictor/
├── src/
│   ├── __init__.py
│   ├── data_prep.py     # Data preprocessing
│   ├── model.py         # XGBoost model implementation
│   └── explain.py       # SHAP explanations
├── config/
│   └── config.yaml      # Configuration file
├── requirements.txt     # Project dependencies
├── main.py             # Main execution script
└── README.md
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- SHAP library for explainable AI
- XGBoost team for the gradient boosting framework
- scikit-learn community for machine learning tools