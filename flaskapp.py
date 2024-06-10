# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import io
import locale
from pathlib import Path

import png
from flask_cors import CORS
from flask import Flask, render_template, send_file, Response, send_from_directory

from generator import *


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

mpl.use('Agg')
# mpl.style.use('seaborn')
locale.setlocale(locale.LC_ALL, '')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
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

    _render = render_template

    @app.route('/')
    def main():
        return _render('index.html')

    @app.route('/app')
    def the_app():
        return _render('app.html')

    @app.route('/WebGL/<path:path>')
    def trial_viewer(path):
        return send_from_directory('static/WebGL', path)

    @app.route('/StreamingAssets/<path:path>')
    def streaming_assets(path):
        return send_from_directory('static/StreamingAssets', path)

    # JSON details
    # ---------------------------------------------------------------------------------------------

    # @app.route('/api/figures/details')
    # def figure_details():
    #     return load_json(figure_details_path())

    @app.route('/api/session/<pid>/details')
    def session_details(pid):
        return load_json(session_details_path(pid))

    @app.route('/api/session/<pid>/trial_details/<int:trial_idx>')
    def trial_details(pid, trial_idx):
        return load_json(trial_details_path(pid, trial_idx))

    # Figures
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/session/<pid>/session_plot')
    def session_overview_plot(pid):
        return send(session_overview_path(pid))

    @app.route('/api/session/<pid>/behaviour_plot')
    def behaviour_overview_plot(pid):
        return send(behaviour_overview_path(pid))

    @app.route('/api/session/<pid>/trial_event_plot')
    def trial_event_overview_plot(pid):
        return send(trial_event_overview_path(pid))

    @app.route('/api/session/<pid>/trial_plot/<int:trial_idx>')
    def trial_overview_plot(pid, trial_idx):
        return send(trial_overview_path(pid, trial_idx))

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")

    app = make_app()
    # to run with SSL, generate certificate with
    # openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
    app.run(ssl_context=('cert.pem', 'key.pem'), port=port)
