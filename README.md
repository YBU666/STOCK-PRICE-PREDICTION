# Stock Price prediction

# Google Stock Price Prediction using LSTM

### Description
This project aims to predict the stock price of Google using Long Short-Term Memory (LSTM) neural networks. We use historical stock price data to train the LSTM model and then make predictions on test data.

### Files
- Google_train_data.csv: CSV file containing historical training data for Google stock.
- Google_test_data.csv: CSV file containing test data for evaluating the model.

### Technologies Used
- Python
- pandas
- NumPy
- Matplotlib
- Scikit-learn
- Keras

### Steps
1. *Data Preprocessing*
   - Read and preprocess historical stock data.
   - Normalize the data using MinMaxScaler.
   - Prepare training data with a time step of 60 days.

2. *Model Creation*
   - Build an LSTM model with four layers and dropout for regularization.
   - Compile the model using the Adam optimizer and mean squared error loss.

3. *Model Training*
   - Train the model using the prepared training data.
   - Evaluate the model's loss over epochs.

4. *Testing and Prediction*
   - Preprocess the test data.
   - Make predictions using the trained LSTM model.
   - Inverse transform the predicted data for visualization.

5. *Visualization*
   - Plot the actual stock prices against predicted prices for evaluation.

### Results
The LSTM model shows promising results in predicting Google's stock prices based on historical data. Visualizations demonstrate the model's performance in capturing price trends.

### Future Improvements
- Fine-tune hyperparameters for better accuracy.
- Explore different LSTM architectures or other neural network models.
- Include additional features for modeling, such as technical indicators or market sentiment analysis.

### References
- [Keras Documentation](https://keras.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
