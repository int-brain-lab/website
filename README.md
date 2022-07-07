# IBL public website prototype

## Development notes

* Install Python requirements
* Launch the development server with `python flaskapp.py` (or `./run.sh`)
* Go to `http://localhost:4321/`


## Deployment on a production server

* Tested on Ubuntu 20.04+
* Create a Python virtual env
* `pip install -r requirements.txt`
* `sudo nano /etc/systemd/system/flaskapp.service` and put:

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

## Unity dev notes

To install the Unity environment you need to clone [vbl-core](https://github.com/VirtualBrainLab/vbl-core) as a git submodule in `Assets/vbl-core`. E.g.:

```
git clone https://github.com/int-brain-lab/website
git submodule add https://github.com/VirtualBrainLab/vbl-core website/UnityMiniBrainClient/Assets/vbl-core
```

The first time you open the project the Unity Editor will install all the dependency packages.

### Addressables

No Addressable assets are used specific to this project. All 3D mesh files are loaded from the [AddressablesStorage](https://github.com/VirtualBrainLab/AddressablesStorage) repo. Data files (e.g. the probe positions csv for the bwm) are currently bundled into the build.

### Unity -> Javascript link

You can add javascript functions to access in this `.jslib` file: [unity_js_link](https://github.com/int-brain-lab/website/blob/main/UnityMiniBrainClient/Assets/Plugins/unity_js_link.jslib). These functions can be called from anywhere in Unity by including a DLL import call referencing the corresponding Javascript functions. Note that **only individual strings or numerical types** can be passed to javascript without dealing directly with the javascript heap.

```
[DllImport("__Internal")]
private static extern void SelectPID(string pid);
```

### Javascript -> Unity link

We exposed a javascript variable `myGameInstance` which can be used to call arbitrary Unity code by using the `SendMessage` function. Floats and strings can be passed as variables.

```
myGameInstance.SendMessage('MyGameObject', 'MyFunction');
myGameInstance.SendMessage('MyGameObject', 'MyFunction', 5);
myGameInstance.SendMessage('MyGameObject', 'MyFunction', 'string');
```

### Build to WebGL

To host the built website you need to put the AddressablesStorage files on the same server or enable cross-origin access on the existing server. You also need to [allow decompression of the Unity files](https://docs.unity3d.com/Manual/webgl-deploying.html).
