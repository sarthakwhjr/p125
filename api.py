from flask import Flask,jsonify,request
from p125 import getpredict
app=Flask(__name__)
@app.route("/predict-digit",methods=["POST"])
def predictdigit():
    Img=request.files.get("digit")
    prediction=getpredict(Img)
    return jsonify({
        'prediction':prediction
    },200)
if __name__=="__main__":
    app.run(debug=True)
