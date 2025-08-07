import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset for encoding consistency
df = pd.read_csv('cars.csv')

# Map owner column to numbers BEFORE get_dummies
df['owner'] = df['owner'].map({
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
})

# Extract features
df['brand'] = df['name'].str.split(' ').str[0]
df['model'] = df['name'].str.split(' ').str[1]
df['car_age'] = 2025 - df['year']
df.drop(['year', 'name'], axis=1, inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'brand', 'model'], drop_first=True)

# Train model
X = df.drop('selling_price', axis=1)
y = np.log1p(df['selling_price'])

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

pred=model.predict(x_test)

print("r2 score is: ",r2_score(y_test,pred))

# Extract options for dropdowns from actual one-hot columns
fuel_options = sorted(df.columns[df.columns.str.startswith("fuel_")].str.replace("fuel_", ""))
seller_options = sorted(df.columns[df.columns.str.startswith("seller_type_")].str.replace("seller_type_", ""))
transmission_options = sorted(df.columns[df.columns.str.startswith("transmission_")].str.replace("transmission_", ""))
brand_options = sorted(df.columns[df.columns.str.startswith("brand_")].str.replace("brand_", ""))
model_options = sorted(df.columns[df.columns.str.startswith("model_")].str.replace("model_", ""))

# Streamlit UI
st.title("🚗 Used Car Selling Price Estimator")

# Sidebar inputs
brand = st.selectbox("Car Brand", brand_options)
model_car = st.selectbox("Car Model", model_options)
kms_driven = st.number_input("Kilometers Driven", value=30000, step=1000)
fuel = st.selectbox("Fuel Type", fuel_options)
seller_type = st.selectbox("Seller Type", seller_options)
transmission = st.selectbox("Transmission", transmission_options)
owner = st.selectbox("Ownership", ['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
car_age = st.slider("Car Age (Years)", min_value=0, max_value=25, value=5)

# Preprocess user input
def preprocess_input():
    input_data = {
        'km_driven': kms_driven,
        'owner': {'Test Drive Car': 0, 'First Owner': 1, 'Second Owner': 2,
                  'Third Owner': 3, 'Fourth & Above Owner': 4}[owner],
        'car_age': car_age
    }

    # One-hot encoding for selected options
    for col in X.columns:
        if col.startswith('fuel_'):
            input_data[col] = 1 if col == f"fuel_{fuel}" else 0
        elif col.startswith('seller_type_'):
            input_data[col] = 1 if col == f"seller_type_{seller_type}" else 0
        elif col.startswith('transmission_'):
            input_data[col] = 1 if col == f"transmission_{transmission}" else 0
        elif col.startswith('brand_'):
            input_data[col] = 1 if col == f"brand_{brand}" else 0
        elif col.startswith('model_'):
            input_data[col] = 1 if col == f"model_{model_car}" else 0
        elif col not in input_data:
            input_data[col] = 0  # other missing features set to 0

    return pd.DataFrame([input_data])

# Predict and display
if st.button("Estimate Price"):
    user_df = preprocess_input()
    log_price = model.predict(user_df)[0]
    predicted_price = np.expm1(log_price)
    st.success(f"💰 Estimated Selling Price: ₹ {predicted_price:,.0f}")
