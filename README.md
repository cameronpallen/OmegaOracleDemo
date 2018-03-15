# OmegaOracleDemo

Demo and funtional test of the Omega Oracle web app. Not connected to the real 
forecast backend for simplicity and to protect sensitive data.

There is a deployment of this app (on a semi-reliable server) 
[__here__](https://oracle.cameronpallen.com/netarts#) which is publicly accessable 
for demonstration purposes.

This app was built to display time series of real time monitoring and forecast
of aragonite saturation state of seawater at shellfish hatcheries.
This seawater chemistry parameter has been shown experimentally to be important
in early development of oyster larvae and thus is of vital interest to hatchery
operators.

The forecasting backend is able to skillfuly predict this parameter based on
tides, wind forcing, and seasonality using a neural network trained on four
years of continuous chemical monitoring.

## Build Info

The only dependencies are the python libraries listed in setup.py and can be
installed with 

```python3 setup.py install``` 

(recomended in a docker container,
virtualenv, etc.), the server
can be run with 

```python3 src/oracle/oracle_server.py```

Default port 7879 can be
updated with the `--port` option. The tempdir option does nothing in this demo
version so do not worry about it, the directory does not have to exist.

Production version requires additional javascript dependencies which are linked
direcly here.

Tested on python 3.6.3
