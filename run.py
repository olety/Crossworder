import os
from flask import *
from task import Crossword
import numpy as np
import time, logging, sys
import multiprocessing

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        with open('static/count.txt', 'r+') as f:
            i = f.readline()
            # TODO: Optimize this?
            if i is '':
                i = 0
            else:
                i = int(i) + 1
            f.seek(0)
            f.write(str(i))
            f.truncate()
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        # proc = multiprocessing.Process(target=
        print(request.form)
        print(bool(int(request.form.get('repeats'))))
        c = Crossword(n=int(request.form.get('rows')),
                      m=int(request.form.get('cols')),
                      repeat_words=bool(int(request.form.get('repeats'))),
                      sort=bool(request.form.get('sort')),
                      maximize_len=bool(int(request.form.get('sort')) - 1))
        c.generate(int(request.form.get('num_words')),
                   save_file='static/{}'.format(i),
                   max_time=int(
                       request.form.get('timeout')),
                   algo=request.form.get('method'))
        return redirect('/{}'.format(i))
    else:
        return render_template('index.html')


@app.route('/<crossword_id>', methods=['GET'])
def show_crossword(crossword_id):
    arr_path = os.path.join(app.static_folder, '{}.npy'.format(crossword_id))
    if os.path.isfile(arr_path):
        arr = np.load(arr_path)
        return render_template('results.html', num_words=arr[6], sid=crossword_id, words=arr[0], board=arr[1][1],
                               setup=arr[2], time=arr[3], max_time=arr[4], algo=arr[5], solved=arr[1][0])
        # return render_template_string(str(arr))
    else:
        return render_template_string('Process timed out.')


if __name__ == '__main__':
    app.run(threaded=False, use_reloader=False)
