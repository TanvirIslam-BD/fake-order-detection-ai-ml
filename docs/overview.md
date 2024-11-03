I've conducted an experiment to train a model and developed a REST API to predict whether an order is fake or genuine. Here’s a summary of the steps I followed:

1.  **Data Collection and Preprocessing**
    
    *   **Data Loading**: Loaded data from an uploaded CSV file.
    *   **Data Cleaning**: Cleaned the price data in the `Amount (Total Price)` and `Coupon amount` columns.
    *   **Feature Extraction**: Extracted date-related features (year, month, day) from the `Date & Time` column.
    *   **Feature and Label Selection**: Selected relevant features (categorical, numeric, and date) and defined the target label.
2.  **Feature Engineering**
    
    *   **Encoding**: Encoded categorical features.
    *   **Scaling**: Scaled numeric features to standardize their values.
    *   **Date and Time Features**: Extracted additional features from date and time information.
    *   **Handling Missing Values**: Imputed missing values to ensure data completeness.
    *   **Transformation Pipeline**: Created a pipeline to automate these transformations.
3.  **Model Selection and Training**
    
    *   **Parameter Extraction**: Retrieved training parameters such as learning rate, maximum iterations, and leaf nodes.
    *   **Data Splitting**: Split the data into training and testing sets.
    *   **Model Creation and Training**: Initialized and trained a model using the Histogram-Based Gradient Boosting Classifier from scikit-learn. (Note: The base model is a decision tree).
4.  **Model Evaluation**
    
    *   **Metrics Calculation**: Evaluated the model on accuracy, precision, recall, F1-score, ROC AUC, and generated a classification report.
5.  **Saving Model and Tracking Training History**
    
    *   Saved the trained model and evaluation metrics for future retraining and tracking the model’s performance over time.
6.  **API Development**
    
    *   Developed a REST API to expose the model for real-time predictions on whether an order is genuine or fake.

### Sample API Requests for Genuine Order Check:
http://127.0.0.1:8080/api/v1/predict
```
Body:
{
    "Amount (Total Price)": 30,
    "Tickets (Quantity)": 2,
    "Coupon amount": 0,
    "Customer Name": "Paresh Khatri",
    "Organisation Name": "Brisbane Football Events Committee",
    "Currency": "AUD",
    "Country": "Australia",
    "Processor": "SecurePay JS SDK",
    "Booking type": "Card",
    "Status": "Active",
    "Payment": "Paid",
    "Date & Time": "10/31/2024  3:34:00 PM"
}
```
```
Response:
{
    "prediction": "Genuine order"
}
```
**************

http://127.0.0.1:8080/api/v1/predict
```
Body:
{
    "Amount (Total Price)": 10,
    "Tickets (Quantity)": 2,
    "Coupon amount": 10,
    "Customer Name": "Jayde Stretton",
    "Organisation Name": "Brisbane Football Events Committee",
    "Currency": "BD",
    "Country": "Bangladesh",
    "Processor": "SecurePay JS SDK",
    "Reconciled": "RECONCILED",
    "Booking type": "Card",
    "Status": "Active",
    "Payment": "Paid",
    "Event Name": "BFEC Carnival 2024",
    "Date & Time": "10/30/2024  9:50:00 PM"
}
```
```
Response:
{
    "prediction": "Order not genuine"
}
```

