import asyncio
import unittest.mock

import oracle.oracle_server as module_ut

_ = unittest.mock.sentinel


def mymock(return_value, spec=None):
    if spec is None:
        spec = []
    return unittest.mock.Mock(return_value=return_value, spec=spec)


class TestOracleService(unittest.TestCase):
    @unittest.mock.patch('oracle.oracle_server.aiohttp', _.aiohttp)
    @unittest.mock.patch('oracle.oracle_server.train_ensemble',
         _.train_ensemble)
    @unittest.mock.patch('oracle.oracle_server.est_Ω', _.est_Ω)
    @unittest.mock.patch('oracle.oracle_server.WebSocket', _.websocket_cls)
    @unittest.mock.patch('oracle.oracle_server.functools', _.functools)
    def test_main(self):
        _.est_Ω.train_ensemble = _.load_model
        _.aiohttp.web = _.aiohttp_web
        _.aiohttp.web.Application = mymock(_.app)
        _.aiohttp.web.run_app = mymock(None)
        _.app.router = _.router
        _.app.router.add_route = mymock(None)
        _.app.on_startup = []
        _.app.on_shutdown = []
        _.websocket_cls.open = mymock(None)
        _.websocket_cls.shutdown = _.websocket_shutdown
        _.functools.partial = mymock(None)
        _.functools.partial.side_effect = (lambda func, *args, **kwargs:
           {_.train_ensemble: _.partial_train,
            _.load_model: _.partial_load_model}[func]
        )
        mock_update_omega = mymock(_.update_job)
        mock_schedule_task = mymock(_.scheduled_job)
        module_ut.ENSEMBLE_SIZE = 2
        with unittest.mock.patch('oracle.oracle_server.update_omega',
               mock_update_omega):
          with unittest.mock.patch('oracle.oracle_server.schedule_task',
                 mock_schedule_task):
            module_ut.main(_.port, 'test_temp')
        _.aiohttp_web.Application.assert_called_once_with()
        self.assertEqual(_.app.on_startup, [_.partial_train, _.scheduled_job])
        self.assertEqual(_.app.on_shutdown, [_.websocket_shutdown])
        mock_schedule_task.assert_called_once_with(_.update_job,
            module_ut.datetime.timedelta(seconds=15))
        _.functools.partial.assert_has_calls([
            unittest.mock.call(_.load_model, weights=[
              'test_temp/oracle_keras_0.h5', 'test_temp/oracle_keras_1.h5']),
            unittest.mock.call(_.train_ensemble, 'test_temp'),
        ], any_order=True)
        self.assertEqual(_.functools.partial.call_count, 2)
        mock_update_omega.assert_called_once_with(_.partial_load_model)
        _.aiohttp.web.run_app.assert_called_once_with(_.app, port=_.port)
        self.assertEqual(_.websocket_cls.app, _.app)

    @unittest.mock.patch('oracle.oracle_server.pandas', _.pandas)
    @unittest.mock.patch('oracle.oracle_server.time', _.time)
    def test_mock_data(self):
        mock_ret = unittest.mock.MagicMock(return_value=None)
        _.pandas.DataFrame = mymock(mock_ret)
        _.pandas.date_range = mymock(_.date_range)
        _.time.sleep = mymock(None)
        ret = module_ut.est_Ω.train_ensemble().predict_live()
        _.time.sleep.assert_called_once_with(10)
        self.assertEqual(mock_ret, ret)
        mock_ret.assert_not_called()
        self.assertEqual(_.pandas.date_range.call_count, 1)

    @unittest.mock.patch('oracle.oracle_server.est_Ω', _.est_Ω)
    def test_train_ensemble(self):
        mock_sleep = mymock(None)
        module_ut.ENSEMBLE_SIZE = 3
        _.est_Ω.train_ensemble = mymock(None)
        async def mock_sleep_async(x):
            mock_sleep(x)
        with unittest.mock.patch('oracle.oracle_server.asyncio.sleep',
            mock_sleep_async):
          asyncio.get_event_loop().run_until_complete(
                  module_ut.train_ensemble('faketempdir', _.app)
          )
        mock_sleep.assert_has_calls([
            unittest.mock.call(1),
            unittest.mock.call(1),
            unittest.mock.call(1),
        ], any_order=True)
        self.assertEqual(mock_sleep.call_count, 3)
        _.est_Ω.train_ensemble.assert_has_calls([
            unittest.mock.call('faketempdir/oracle_keras_0.h5'),
            unittest.mock.call('faketempdir/oracle_keras_1.h5'),
            unittest.mock.call('faketempdir/oracle_keras_2.h5'),
        ], any_order=True)
        self.assertEqual(_.est_Ω.train_ensemble.call_count, 3)


if __name__ == '__main__':
    unittest.main()
