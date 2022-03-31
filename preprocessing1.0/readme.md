
#flask set DEBUG mode and run
- powershell
  - $env:FLASK_APP = "helloworld.py"
  - $env:FLASK_ENV = "development"
  - flask run
- cmd
  - set FLASK_DEBUG=1
  - set FLASK_APP=app
  - flask run