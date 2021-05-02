from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)


@app.route("/")
def home():
  # Inline html
  return 'Hello! <h1>Heading!</h1>'

@app.route("/hello")
def hello():
  # html file in templates folder
  return render_template("page.html")

@app.route("/dynamic")
def dynamic():
  # html file in templates folder
  return render_template("dynamic.html", name='world')

@app.route("/<placeholder>")
def catch(placeholder):
  return f'You are on page {placeholder}!'

@app.route("/admin")
def admin():
  # Redirect to 'catch' method
  return redirect(url_for('catch', placeholder='foobar'))

@app.route("/slash/")
def slash(): 
  return 'Works with and without slash in the url'

@app.route("/inline_python")
def py_inline():
  # html file in templates folder
  return render_template("inline_python.html", mylist=["a", "b", "c"])

if __name__ == "__main__":
  app.run()

