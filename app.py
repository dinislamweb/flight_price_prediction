import gradio as gr
import pandas as pd
import pickle
import os

# load the model (the saved model is a pipeline with preprocessing)
try:
    if not os.path.exists("best_rf_model.pkl"):
        raise FileNotFoundError("Model file 'best_rf_model.pkl' not found.")
    with open("best_rf_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# main logic
def predict_flight_price(airline, source_city, departure_time, stops, arrival_time,
                         destination_city, flight_class, duration, days_left):
    
    if model is None:
        return "Error: Model not loaded. Please ensure 'best_rf_model.pkl' exists."
    
    try:
        # Create input DataFrame with exact column names and order as in training
        input_df = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'destination_city': [destination_city],
            'departure_time': [departure_time],
            'arrival_time': [arrival_time],
            'stops': [stops],  # Keep as string: 'zero', 'one', 'two_or_more'
            'class': [flight_class],
            'duration': [float(duration)],
            'days_left': [float(days_left)]
        })
        
        # prediction using the pipeline (which includes preprocessing)
        prediction = model.predict(input_df)[0]
        return f"Predicted Flight Price: ₹{prediction:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# inputs
inputs = [
    gr.Dropdown(choices=['Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet'], label='Airline'),
    gr.Dropdown(choices=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], label='Source City'),
    gr.Dropdown(choices=['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night'], label='Departure Time'),
    gr.Dropdown(choices=['zero', 'one', 'two_or_more'], label='Stops'),
    gr.Dropdown(choices=['Morning', 'Early_Morning', 'Evening', 'Night', 'Afternoon', 'Late_Night'], label='Arrival Time'),
    gr.Dropdown(choices=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'], label='Destination City'),
    gr.Radio(choices=['Economy', 'Business'], label='Class'),
    gr.Number(label="Duration (hours)", minimum=0),
    gr.Number(label="Days left before departure", minimum=0)
]

# interface
app = gr.Interface(
    fn=predict_flight_price,
    inputs=inputs,
    outputs="text",
    title="Flight Price Predictor"
)

# launch the app
if __name__ == "__main__":
    app.launch(share=True)
