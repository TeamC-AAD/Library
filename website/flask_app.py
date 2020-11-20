from flask import Flask, Response, render_template

app = Flask(__name__)

@app.route('/')
def mainpage():
    return render_template("index.html")

@app.route('/report')
def reportpage():
    return render_template("report.html")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
