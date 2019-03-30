# Import libraries
import numpy as np
from flask import Flask, request, jsonify

from sklearn.externals import joblib
app = Flask(__name__)
# Load the model
model = joblib.load(open("multiple.linear.pkl","rb"))
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    
    ran_data = [data['Grade'], data['Latitude'], data['SqureFeet'] , data['AboveGround'], data['WaterFront'], data['View']]
    ran_data_arr = np.array(ran_data)
    ran_data_num = ran_data_arr.reshape(1,-1)
    prediction = model.predict(ran_data_num)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port =5050)

