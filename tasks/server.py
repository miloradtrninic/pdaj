from datetime import datetime
import os
import socket
import subprocess
import time
import csv

from celery import chain, chord
from celery.exceptions import Reject
import numpy as np
import tables as tb

from ..app import app
from .worker import simulate_pendulum


## Monitoring tasks

@app.task
def monitor_queues(ignore_result=True):
    server_name = app.conf.MONITORING_SERVER_NAME
    server_port = app.conf.MONITORING_SERVER_PORT
    metric_prefix = app.conf.MONITORING_METRIC_PREFIX

    queues_to_monitor = ('server', 'worker')
    
    output = subprocess.check_output('rabbitmqctl -q list_queues name messages consumers', shell=True)
    lines = (line.split() for line in output.splitlines())
    data = ((queue, int(tasks), int(consumers)) for queue, tasks, consumers in lines if queue in queues_to_monitor)

    timestamp = int(time.time())
    metrics = []
    for queue, tasks, consumers in data:
        metric_base_name = "%s.queue.%s." % (metric_prefix, queue)

        metrics.append("%s %d %d\n" % (metric_base_name + 'tasks', tasks, timestamp))
        metrics.append("%s %d %d\n" % (metric_base_name + 'consumers', consumers, timestamp))

    sock = socket.create_connection((server_name, server_port), timeout=10)
    sock.sendall(''.join(metrics))
    sock.close()


## Recording the experiment status

def get_experiment_status_filename(status):
    return os.path.join(app.conf.STATUS_DIR, status)

def get_experiment_status_time():
    """Get the current local date and time, in ISO 8601 format (microseconds and TZ removed)"""
    return datetime.now().replace(microsecond=0).isoformat()


@app.task
def record_experiment_status(status):
    with open(get_experiment_status_filename(status), 'w') as fp:
        fp.write(get_experiment_status_time() + '\n')


## Seeding the computations

@app.task
def simulate_pendulum(ignore_result=True):
    if os.path.exists(get_experiment_status_filename('started')):
        raise Reject('Computations have already been seeded!')

    record_experiment_status.si('started').delay()
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    THETA_RESOLUTION = app.conf.THETA_RESOLUTION
    TIME_MAX = app.conf.TIME_MAX
    DTIME = app.conf.DTIME

    chord(
        (simulate_pendulum(L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init)
         for (L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init) in parametar_sweep(L1, L2, m1, m2, TIME_MAX, DTIME, THETA_RESOLUTION)),
        save_pendulum_point.s()
    ).delay()

def parametar_sweep(L1, L2, m1, m2, tmax, dt, theta_resolution):
    for theta1_init in np.linspace(0, 2 * np.pi, theta_resolution):
        for theta2_init in np.linspace(0, 2 * np.pi, theta_resolution):
            return L1, L2, m1, m2, tmax, dt, theta1_init, theta2_init


## Storing the computed integral tables

@app.task
def save_pendulum_point(results):
    with open("results.csv", 'w') as resultsfile:
        resultsfile.write(results)
