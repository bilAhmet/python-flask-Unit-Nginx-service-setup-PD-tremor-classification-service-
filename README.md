# python-flask-Unit-Nginx-service-setup-PD-tremor-classification-service-


# Steps of service setup [Python, ubuntu server, Nginx, Unit, Flask]


## Required for downloading in the virtual environment: these python libraries:
    • Flask 1.1.2
    • flask_cors 3.0.9
    • numpy 1.18.5
    • matplotlib 
    • pandas 1.1.3
    • scipy 1.5.2
    • datetime
    • tensorflow 1.13.1, pip install tensorflow==1.13.1
    • sklearn 0.23.2

## how to proceed
### install Unit; steps in the link below :
https://unit.nginx.org/installation/#ubuntu_1804

### In console:

```bash
$cd /path/to/app/
$python3 -m venv name_of_venv
$source  name_of_venv/bin/activate
$pip install (all required libraries above)
$deactivate
```


### Put the wsgi.py and all required project files in the app path

### Set installation path permissions to secure access, for example:
```bash
chown -R unit_user:unit_group /path/to/app/
```

### create config.json in the app path and fill the below in it:
```json
{
    "listeners": {
        "*:8000": {
            "pass": "applications/flask_app"
        }
    },

    "applications": {
        "flask_app": {
            "type": "python 3.6",
            "user": "unit_user",
            "group": "unit_group",
            "path": "/path/to/app/",
            "home": "/path/to/app/name_of_venv/",
            "module": "wsgi"
        }
    }
}
```

### Upload the updated configuration:
```bash
sudo curl -X PUT --data-binary @config.json --unix-socket  /var/run/control.unit.sock http://localhost/config/
```

### Verify the port:
```bash
sudo ufw allow  (port_number)
```

### For restarting services of Unit:
```bash
systemctl restart unit.service
```


### For log information:
```bash
sudo more /var/log/unit.log
```
