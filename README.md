# OmegaOracleDemo
Demo and funtional test of the Omega Oracle web app. Not connected to the real forecast backend for simplicity and to protect sensitive data.

This app was built to display time series of real time monitoring and forecast
of aragonite saturation state of seawater at shellfish hatcheries.
This seawater chemistry parameter has been shown experimentally to be important
in early development of oyster larvae and thus is of vital interest to hatchery
operators.

The forecasting backend is able to skillfuly predict this parameter based on
tides, wind forcing, and seasonality using a neural network trained on four
years of continuous chemical monitoring.

There is a deployment of this app (on a semi-reliable server) [here](https://oracle.cameronpallen.com/netarts#) which is publicly accessable for demonstration purposes.
