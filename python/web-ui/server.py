import argparse
from markupsafe import escape
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
  return "Hello World!"

@app.route('/compute/<op>')
def compute(op):

  def operands(n):
    l = [0] * n
    err = ""

    params = request.args.to_dict()
    if len(params) != n:
      err = f"Error: expecing '{n}' operands, found '{len(params)}'"
    else:
      try:
        l = [int(val) for val in params.values()]
      except:
        err = f"Error: converting to int '{params.values()}'"

    return tuple([err] + l)

  match op:
    case 'neg':
      error, a = operands(1)
      result = -1 * a if error == "" else error

    case 'add':
      error, a, b = operands(2)
      result = a + b if error == "" else error

    case 'sub':
      error, a, b = operands(2)
      result = a - b if error == "" else error

    case 'mul':
      error, a, b = operands(2)
      result = a * b if error == "" else error

    case _:
      result = f"<i>Error: Unknown opration '{escape(op)}'<i>"

  return str(result)


@app.route('/form', methods=['GET', 'POST'])
def index():
  control = []
  experiment = []

  if request.method == 'POST':
    query = request.form['query']
    control = ['Hello control', 'abc foo bar', 'def last']
    experiment = ['Hello experiment', 'xyz baz', 'pqr data']
    print(query)

  return render_template('index.html', control=control, experiment=experiment)


def main():
  # Get commandline arguments
  parser = argparse.ArgumentParser(description='Web server.')
  parser.add_argument('-p', '--port', help='Port')
  args = parser.parse_args()

  app.run(debug=True, port=args.port)


if __name__ == '__main__':
  main()

