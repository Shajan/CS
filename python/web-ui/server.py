import argparse

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
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

