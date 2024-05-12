# Stock Price Prediction

## Google Stock Price Prediction using LSTM

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
   - Read and preprocess historical training data.
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

6. *Results*
   The LSTM model shows promising results in predicting Google's stock prices based on historical data. Visualizations demonstrate the model's performance in capturing price trends.

7. *Future Improvements*
   - Fine-tune hyperparameters for better accuracy.
   - Explore different LSTM architectures or other neural network models.
   - Include additional features for modeling, such as technical indicators or market sentiment analysis.

8. *References*
   - [Keras Documentation](https://keras.io/)
   - [Scikit-learn Documentation](https://scikit-learn.org/)
   - [Matplotlib Documentation](https://matplotlib.org/)

## Tesla Stock Price Prediction using Linear Regression

### Description
This project predicts the stock price of Tesla using linear regression. We split the data into training and testing sets, preprocess the data, and build a linear regression model to make predictions.

### Files
- tesla.csv: CSV file containing historical stock price data for Tesla.

### Technologies Used
- Python
- pandas
- NumPy
- Matplotlib
- Scikit-learn

### Steps
1. *Data Preprocessing*
   - Read and preprocess historical stock price data.
   - Normalize and scale the data using StandardScaler.
   - Split the data into training and testing sets.

2. *Model Creation*
   - Build a linear regression model using Scikit-learn.

3. *Model Training*
   - Train the linear regression model using training data.

4. *Testing and Prediction*
   - Make predictions using the trained linear regression model on test data.

5. *Visualization*
   - Plot actual vs. predicted values for training and testing datasets.

6. *Model Evaluation*
   - Calculate evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Median Absolute Error, and Explained Variance.

7. *Results*
   The linear regression model demonstrates good performance in predicting Tesla's stock prices. Evaluation metrics provide insights into the model's accuracy and performance.

8. *Future Improvements*
   - Experiment with different regression algorithms for comparison.
   - Incorporate additional features for modeling, such as economic indicators or news sentiment analysis.

9. *References*
   - [Scikit-learn Documentation](https://scikit-learn.org/)
   - [Matplotlib Documentation](https://matplotlib.org/)
