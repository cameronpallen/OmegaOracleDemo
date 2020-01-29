'''
The aiohttp http endpoint serves html (+ css, etc) to the client.
The client (through javascript in index.html) then opens a websocket
connection with the aiohttp server. The server keeps track of each of these
websocket connections with all connected clients. The server is periodically
updating its prediction as it downloads more weather predicion data from the
web and receives new telemetry from the chemical monitoring instrument at the
oyster hatchery. Each time these predicionts and measurements are updated the
server uses the websocket connections to send a new packet of data to each
of the connected clients so their plots can be udpated.
'''
import argparse
import asyncio
import concurrent.futures
import datetime
import functools
import json
import logging
import os
import sys
import threading

import aiohttp.web
import numpy
import pandas

ENSEMBLE_SIZE = 10


#import wcsh.omeof as est_Ω
# est_Ω does all of the work of making actual predictions. For this
# demo/testing version of the front end we mock that out with fake data.
import mock
import time
est_Ω = mock.Mock()
def mock_predict():
    # Simulate cpu blocking to demonstrate the server remains responsive.
    time.sleep(3)
    N = 24 * 4 * 14
    # return some fake data.
    ret = pandas.DataFrame(
      {
        'Ω': (1.5
           + .5 * numpy.sin(numpy.arange(N) * (2 * numpy.pi) * 14 / N)
           + .4 * numpy.sin(numpy.arange(N) * (2 * numpy.pi) * 28 / N)
           + .2 * numpy.cos(numpy.arange(N) * (2 * numpy.pi) * 20 / N)
           # Include some randomness so we can see the data change
           #    as the app is running
           + .15 * numpy.random.rand(N)
        ),
        'Ω_est': (1.5
           + .5 * numpy.sin(numpy.arange(N) * (2 * numpy.pi) * 14 / N)
           + .35 * numpy.sin(numpy.arange(N) * (2 * numpy.pi) * 28 / N)
        ),

      },
      pandas.date_range(
          start=datetime.datetime.now() - datetime.timedelta(7),
          periods=N, freq='15T'
      )
    )
    ret.loc[datetime.datetime.now():, 'Ω'] = numpy.nan
    return ret
est_Ω.train_ensemble.return_value.predict_live = mock_predict


def main(port, tempdir):
    logging.root.setLevel(logging.INFO)
    app = aiohttp.web.Application()
    # Endpoints for the static html, css, and image files.
    app.router.add_route(
            'GET', r'/netarts{page:|/|/[^{}/]+\.(css|js|html|png|ico)}',
            index(os.path.dirname(os.path.abspath(__file__)))
    )
    app.router.add_route(
            'GET', r'/{page:|[^{}/]+\.(css|js|html|png|ico)}',
            index(os.path.dirname(os.path.abspath(__file__)))
    )
    # Endpoint for webclient to start a websocket connection.
    app.router.add_route(
            'GET', r'/oracle_ws/netarts/', WebSocket.open()
    )
    WebSocket.app = app
    # Ensure ensemble training is done before startup.
    app.on_startup.append(functools.partial(train_ensemble, tempdir))
    # Paramterize the scheduled forecast updates with the locations of the
    #   cached model parameterization
    load_model = functools.partial(
        est_Ω.train_ensemble,
        weights=[
            os.path.join(tempdir, 'oracle_keras_{}.h5'.format(i))
            for i in range(ENSEMBLE_SIZE)
        ]
    )
    update_job = update_omega(load_model)
    # Begin scheduled updates of model prediction
    app.on_startup.append((schedule_task(update_job,
        datetime.timedelta(seconds=5))))
    app.on_shutdown.append(WebSocket.shutdown)
    # Start serving the app.
    aiohttp.web.run_app(app, port=port)


async def train_ensemble(tempdir, app):
    """
    Ensure that all the prediction ensamble members are trained and cached to
    disk at tempdir.
    """
    logging.info('Ensure models trained. tempdir: {}'.format(tempdir))
    for i in range(ENSEMBLE_SIZE):
        await _train_ensemble_member(tempdir, i)
    logging.info('Models trained.')


async def _train_ensemble_member(tempdir, i):
    est_Ω.train_ensemble(os.path.join(tempdir, 'oracle_keras_{}.h5'.format(i)))
    await asyncio.sleep(.1)


def update_omega(load_model):
    '''
    Run a new model prediction based on updated weather data and push the new
    prediction messages to queues to transmit to all connected clients.

    Takes a function which loads the model parameters from disk each time the
    prediction is run so that the model can be updated on the fly.

    returns a function which called with no parameters does all the updates.
    '''
    def do_prediction():
        logging.info('do omega estimate update')
        # Get the preduction result
        df = load_model().predict_live()
        # Convert data to format expected by frontend
        message = json.dumps([
            json.loads(x.to_json(date_format='iso'))
            for _, x in
            df.reset_index().rename(
              columns={'index': 'time', 'Ω': 'omega', 'Ω_est': 'omega_est'}
            ).dropna(0, 'all').iterrows()
        ], sort_keys=True)
        logging.info('update done. writting message to queues....')
        # Send message to all attached clients
        WebSocket.put_message(message)
        logging.info('message written to queues.')
    return do_prediction


def _schedule_task(loop, task, interval, executor):
    '''
    Keeps track of the timing of using executor to do task every interval often
    '''
    def make_checker(next_check_schedule):
        # Wrapper function to allow the callback structure.
        # Call the input fuction when done to schedule the next execution.
        async def check_schedule(when):
            # Does the actual schedule checking logic and excecution of task
            try:
                if datetime.datetime.utcnow() > when:
                    # Interval expired, execute the task.
                    await loop.run_in_executor(executor, task)
                    # Sucessful, set the time to excecute the task next.
                    when += interval
                    logging.info('next update: {}'.format(when.strftime(
                        '%Y-%m-%d %H:%M:%S')))
                else:
                    # Short wait before checking the schedule again.
                    await asyncio.sleep(1)
            finally:
                # Always schedule another check, even if there is an exception.
                next_check_schedule(when)
        return check_schedule
    return make_checker


def schedule_task(task, interval, delay=None):
    '''
    Excecute the function task periodically with interval interval
      (a timedelta) between excecutions and an initial delay (also a timedelta)

    Returns a fuction the meets the spec required as an argument
      to app.on_startup

    Contains all the really nasty async callback logic to keep that out of
      the rest of the functions.

    '''
    # Excecutor used to run the cpu-intensive predictions without making the
    # server block.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if delay is None:
        delay = datetime.timedelta(seconds=1)
    async def schedule_task_async(app):
        scheduler = _schedule_task(app.loop, task, interval, executor)
        # Create the callback structure using the scheduler.
        init_schedule = lambda init_schedule_next: (
            lambda when: asyncio.ensure_future(
                scheduler(init_schedule_next(init_schedule_next))(when),
                loop=app.loop
            )
        )
        # Start the schedule with first execution at now + delay.
        init_schedule(init_schedule)(datetime.datetime.utcnow() + delay)
    return schedule_task_async


def _read_file(directory, page):
    fname = os.path.join(directory, page)
    with open(fname, 'rb') as fh:
        ret = fh.read()
    return ret


def index(directory):
    ''' Serve static files from the current directory. '''
    async def _index(request):
        page = (request.match_info.get('page').strip('/') or 'index.html').strip('/')
        logging.info('load page: [{}]'.format(
            os.path.join(directory, page))
        )
        if page not in os.listdir(directory):
            raise ValueError('PAGE NOT FOUND')
        response = dict(body=_read_file(directory, page))
        if page.endswith('.css'):
            response['content_type'] = 'text/css'
        if page.endswith('.html'):
            response['content_type'] = 'text/html'
        return aiohttp.web.Response(**response)
    return _index


class WebSocket(object):
    '''
    Class to manage all of the servers open websockets with various
    active web clients.
    '''
    last_message = None # Use this to send to new client connections
    # This lock is probably unecessary, assuming the assignment is atomic,
    #   but may as well be extra safe.
    last_message_lock = threading.Lock()
    queues = [] # list of all active asnycio queues
    wss = [] # list of active websockets
    queues_list_lock = threading.Lock()
    app = None

    def __init__(self, *args, **kwargs):
        self._queue = asyncio.Queue()
        with WebSocket.queues_list_lock:
            WebSocket.queues.append(self._queue)
        self._task = None
        super().__init__(*args, **kwargs)

    @classmethod
    def put_message(cls, message):
        '''
        Queue a message to be sent out to all acive clients.
        '''
        return asyncio.ensure_future(
                WebSocket._put_message(message),
                loop=cls.app.loop
        )

    @classmethod
    async def _put_message(cls, message):
        with cls.last_message_lock:
            cls.last_message = message
        with cls.queues_list_lock:
            for queue in cls.queues:
                await queue.put(message)

    async def _proc_queue(self, ws):
        try:
            message = await self._queue.get()
            await ws.send_str(message)
            self._queue.task_done()
        finally:
            if ws.closed:
                self._on_close(ws)
            else:
                self._sched_proc_queue(ws)

    def _sched_proc_queue(self, ws):
        '''
        Send enqueued messages to websocket ws until ws closes.
        '''
        self._task = asyncio.ensure_future(
                self._proc_queue(ws), loop=WebSocket.app.loop
        )

    def _on_close(self, ws):
        '''
        cleanup when websocket closes
        '''
        if self._task:
            self._task.cancel()
        with WebSocket.queues_list_lock:
            if self._queue in WebSocket.queues:
                WebSocket.queues.remove(self._queue)
            if ws in WebSocket.wss:
                WebSocket.wss.remove(ws)

    @classmethod
    def open(cls):
        '''start new websocket connection for a new client.'''
        async def handler(request):
            # Create the new WebSocket instance.
            instance = cls()
            with cls.last_message_lock:
                # Send most recent data immedietly.
                if cls.last_message is not None:
                    await instance._queue.put(cls.last_message)
            ws = aiohttp.web.WebSocketResponse()
            await ws.prepare(request)
            with WebSocket.queues_list_lock:
                WebSocket.wss.append(ws)
            try:
                # Begin processing messages from the queue.
                instance._sched_proc_queue(ws)
                # Send messages every 20 seconds for logging and to keep the
                #   websocket conneciton alive.
                await ws.send_str('ping')
                async for msg in ws:
                    # Every time client responds with a ping wait ten seconds
                    #   and then send another ping to keep connection alive
                    logging.info(msg)
                    await asyncio.sleep(10)
                    ws.send_str('ping')
            finally:
                instance._on_close(ws)
                return ws
        return handler

    @classmethod
    async def shutdown(cls, *args, **kwargs):
        ''' Close all client connections, sending an appropriate message to the
        web clients.'''
        with cls.queues_list_lock:
            #for ws in cls.wss:
            #    await ws.close(code=999, message='server shutdown')
            await asyncio.gather(*[
                 ws.close(code=999, message='server shutdown')
                 for ws in cls.wss
            ], loop=kwargs.get('loop', None))


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=7879, type=int)
    parser.add_argument('--tempdir', default='/tmp/oracle/')
    args = parser.parse_args(args)
    return args.port, args.tempdir


if __name__ == '__main__':
    main(*parse_args(sys.argv[1:]))
