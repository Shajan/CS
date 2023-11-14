import argparse

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def get_input():
  return render_template('question.html')


@app.route('/result', methods=['POST'])
def find_ann():
  answer = request.form['question']
  return answer


@app.route('/test')
def test():
  return 'Hello, World!'


def main():
  # Get commandline arguments
  parser = argparse.ArgumentParser(description='Web server.')
  parser.add_argument('-p', '--port', help='Port')

  args = parser.parse_args()

  app.run(debug=True, port=args.port)
  pass


if __name__ == '__main__':
  main()

