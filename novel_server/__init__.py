from flask import Flask
from views import create_ill, step2_views, step3_views


app = Flask(__name__)

app.register_blueprint(create_ill.bp)
app.register_blueprint(step2_views.bp)
app.register_blueprint(step3_views.bp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
