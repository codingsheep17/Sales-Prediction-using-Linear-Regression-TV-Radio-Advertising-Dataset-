import pandas as pd
import pickle

with open("ad_model.pkl", 'rb') as f:
    model = pickle.load(f)
    
def convert_input(tv, radio):
    #convert the tv and radio in 1 row and 2 columns shape
    custom_input = pd.DataFrame({
            "TV":[tv],
            "Radio":[radio]
        })
    prediction = model.predict(custom_input)
    return prediction[0]

my_tv = 150
my_radio = 30
predicted_sales = convert_input(my_tv, my_radio)
print(f"Predicted Sales:{predicted_sales:.2f}")