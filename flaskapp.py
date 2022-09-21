# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import io
import locale
import logging
import os.path as op
from pathlib import Path

import png
from flask_cors import CORS
from flask_caching import Cache
from flask import Flask, render_template, send_file, Response

from generator import *


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

mpl.use('Agg')
# mpl.style.use('seaborn')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------------------

_logger_fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = {'D': '90', 'I': '0', 'W': '33', 'E': '31'}.get(record.levelname, '7')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


logger = logging.getLogger('ibl_website')
logger.setLevel(logging.DEBUG)


def add_default_handler(level='DEBUG', logger=logger):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
PORT = 4321
DEFAULT_PID = '7d999a68-0215-4e45-8e6c-879c6ca2b771'


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        self.__dict__ = self
        super().__init__(*args, **kwargs)


def to_png(arr):
    p = png.from_array(arr, mode="L")
    b = io.BytesIO()
    p.write(b)
    b.seek(0)
    return b


def send_image(img):
    return send_file(to_png(img), mimetype='image/png')


def send_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def send_png_bytes(btes):
    buf = io.BytesIO(btes)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def send(path):
    if path.exists():
        return send_file(path)
    else:
        logger.error(f"path {path} does not exist")
        return Response(status=404)


# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------

def get_sessions():
    return [{'pid': pid} for pid in get_pids()]


def get_js_context():
    return {}


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------

def make_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    CORS(app, support_credentials=True)

    # ---------------------------------------------------------------------------------------------
    # Entry points
    # ---------------------------------------------------------------------------------------------

    def _render(fn):
        return render_template(
            fn,
            sessions=get_sessions(),
            default_pid=DEFAULT_PID,
            js_context=get_js_context(),
        )

    @app.route('/')
    def main():
        return _render('index.html')

    @app.route('/app')
    def the_app():
        return _render('app.html')

    # JSON details
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/session/<pid>/details')
    def session_details(pid):
        return load_json(session_details_path(pid))

    @app.route('/api/session/<pid>/trial_details/<int:trial_idx>')
    def trial_details(pid, trial_idx):
        return load_json(trial_details_path(pid, trial_idx))

    @app.route('/api/session/<pid>/cluster_details/<int:cluster_idx>')
    def cluster_details(pid, cluster_idx):
        return load_json(cluster_details_path(pid, cluster_idx))

    # Figures
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/session/<pid>/session_plot')
    def session_overview_plot(pid):
        return send(session_overview_path(pid))

    @app.route('/api/session/<pid>/trial_plot/<int:trial_idx>')
    def trial_overview_plot(pid, trial_idx):
        return send(trial_overview_path(pid, trial_idx))

    @app.route('/api/session/<pid>/cluster_plot/<int:cluster_idx>')
    def cluster_overview_plot(pid, cluster_idx):
        return send(cluster_overview_path(pid, cluster_idx))

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")

    app = make_app()
    app.run('0.0.0.0', port=port)
