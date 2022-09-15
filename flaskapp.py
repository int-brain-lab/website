# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import io
import locale
import logging
from pathlib import Path
import png

from flask_cors import CORS
from flask_caching import Cache
from flask import Flask, render_template, send_file, Response

from generator import *


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('ibl_website')
mpl.use('Agg')
# mpl.style.use('seaborn')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data'
PORT = 4321


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
