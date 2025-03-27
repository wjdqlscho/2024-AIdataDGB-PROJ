import pymongo
import certifi
from flask import Flask, jsonify

app = Flask(__name__)

MONGO_URI = "추후 변경"
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client['project_db']
collection = db['input_data']

@app.route('/results', methods=['GET'])
def get_results():
    data = list(collection.find({}, {'_id': 0}))
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
