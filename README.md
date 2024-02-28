# Generic visualization website

Generic version of the [International Brain Lab visualization website](https://viz.internationalbrainlab.org/app).

## Quick start instructions

This repository contains example data corresponding to a single session, a single cluster, and a single trial.

1. Create a local pair of SSL keys to run the website locally on HTTPS:

    ```
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
    ```

2. Install the Python requirements:

    ```
    pip install -r requirements.txt
    ```

3. Launch the development server:

    ```
    bash run.sh
    ```

4. Browse to `https://127.0.0.1:4321/`.

## How to integrate your own data

- The website presents five figures:

  - Session overview : figures 1 and 2;
  - Single trial overview : figures 3 and 4;
  - Single cluster overview : figure 5.

- `static/cache/`: contains your figures and metadata.
- `static/cache/figures.json`: contains the panel information (corner coordinates, letters, legends) of each figure.
- `static/<PID>`: contains the figures and metadata of a given probe insertion. **PID** is the probe unique identifier, for example `1ab86a7f-578b-4a46-9c9c-df3be97abcca`.
- This subfolder contains the figures:

    - `overview.png`: this is **Figure 1**;
    - `behaviour_overview.png`: this is **Figure 2**;
    - `trial-dddd.png`: this is **Figure 3** for trial #dddd;
    - `trial_overview.png`: this is **Figure 4**;
    - `cluster-dddd.png`: this is **Figure 5** for cluster #dddd.

- This subfolder also contains the metadata:

    - `session.json`: the session metadata, including the list of clusters, trials, and other metadata used in the session search bar;
    - `trial-dddd.json`: the trial metadata for cluster #dddd, used in **Figure 3**;
    - `trial_intervals.csv`: the time intervals of each trial, used in **Figure 4**;
    - `cluster-dddd.json`: the cluster metadata for cluster #dddd, used in **Figure 5**;
    - `cluster_pixels.csv`: the cluster coordinates in **Figure 5** (relative to the figure size in pixels)

To integrate your own data, generate your figures and metadata and organize your files accordingly in `static/cache/`.

Finally, go to `flaskapp.py`, and set `DEFAULT_PID` to the unique ID of the default probe insertion you want to load.


## Deployment notes on a production server

* Tested on Ubuntu 20.04+
* Create a Python virtual env
* `pip install -r requirements.txt`
* `sudo nano /etc/systemd/system/flaskapp.service` and put:

```
[Unit]
Description=My website
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/website/
Environment="PATH=/home/ubuntu/website/bin"
ExecStart=sudo /home/ubuntu/website/bin/python flaskapp.py --port 80

[Install]
WantedBy=multi-user.target
```
