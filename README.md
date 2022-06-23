# IBL public website prototype


### Install notes (to be completed)

* Tested on Ubuntu 20.04
* Create a Python virtual env
* `pip install -r requirements.txt`
* For deployment, `sudo nano /etc/systemd/system/flaskapp.service` and put:

```
[Unit]
Description=IBL website
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
