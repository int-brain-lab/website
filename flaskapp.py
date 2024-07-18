# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import io
import locale
from pathlib import Path

import png
from flask_cors import CORS
from flask import Flask, render_template, send_file, Response, send_from_directory, request

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
        spikesorting = request.args.get('spikesorting', 'ss_original')
        return _render('app.html', spikesorting=spikesorting)

    @app.route('/WebGL/<path:path>')
    def trial_viewer(path):
        return send_from_directory('static/WebGL', path)

    @app.route('/StreamingAssets/<path:path>')
    def streaming_assets(path):
        return send_from_directory('static/StreamingAssets', path)

    # JSON details
    # ---------------------------------------------------------------------------------------------

    # @app.route('/api/<spikesorting>/figures/details')
    # def figure_details():
    #     return load_json(figure_details_path())

    @app.route('/api/<spikesorting>/session/<pid>/details')
    def session_details(spikesorting, pid):
        return load_json(session_details_path(spikesorting, pid))

    @app.route('/api/<spikesorting>/session/<pid>/trial_details/<int:trial_idx>')
    def trial_details(spikesorting, pid, trial_idx):
        return load_json(trial_details_path(spikesorting, pid, trial_idx))

    @app.route('/api/<spikesorting>/session/<pid>/cluster_details/<int:cluster_idx>')
    def cluster_details(spikesorting, pid, cluster_idx):
        return load_json(cluster_details_path(spikesorting, pid, cluster_idx))

    @app.route('/api/<spikesorting>/session/<pid>/cluster_plot_from_xy/<int:cluster_idx>/<float(signed=True):x>_<float(signed=True):y>/<int:qc>')
    def cluster_from_xy(spikesorting, pid, cluster_idx, x, y, qc):
        cluster_idx, idx = get_cluster_idx_from_xy(spikesorting, pid, cluster_idx, x, y, qc)
        return {
            "idx": int(idx),
            "cluster_idx": int(cluster_idx),
        }

    # Figures
    # ---------------------------------------------------------------------------------------------

    @app.route('/api/<spikesorting>/session/<pid>/session_plot')
    def session_overview_plot(spikesorting, pid):
        return send(session_overview_path(spikesorting, pid))

    @app.route('/api/<spikesorting>/session/<pid>/behaviour_plot')
    def behaviour_overview_plot(spikesorting, pid):
        return send(behaviour_overview_path(spikesorting, pid))

    @app.route('/api/<spikesorting>/session/<pid>/trial_event_plot')
    def trial_event_overview_plot(spikesorting, pid):
        return send(trial_event_overview_path(spikesorting, pid))

    @app.route('/api/<spikesorting>/session/<pid>/trial_plot/<int:trial_idx>')
    def trial_overview_plot(spikesorting, pid, trial_idx):
        return send(trial_overview_path(spikesorting, pid, trial_idx))

    @app.route('/api/<spikesorting>/session/<pid>/cluster_plot/<int:cluster_idx>')
    def cluster_overview_plot(spikesorting, pid, cluster_idx):
        return send(cluster_overview_path(spikesorting, pid, cluster_idx))

    @app.route('/api/<spikesorting>/session/<pid>/cluster_qc_plot/<int:cluster_idx>')
    def cluster_qc_overview_plot(spikesorting, pid, cluster_idx):
        return send(cluster_qc_overview_path(spikesorting, pid, cluster_idx))

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
    #app.run(ssl_context=('cert.pem', 'key.pem'), port=port)
    app.run()
