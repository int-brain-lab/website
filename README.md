# Generic visualization website

Generic version of the [International Brain Lab visualization website](https://viz.internationalbrainlab.org/app).

## Instructions

1. Create a local pair of SSL keys to run the website locally on HTTPS:

    ```
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
    ```

2. Install the Python requirements:
    ```
    pip install -r requirements.txt
    ```


. Launch the development server:

    ```
    bash run.sh
    ```

. Browse to `https://127.0.0.1:4321/`.



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
