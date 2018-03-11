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
        mock_update_omega.assert_called_once_with(_.partial_load_model)
        _.aiohttp.web.run_app.assert_called_once_with(_.app, port=_.port)
        self.assertEqual(_.websocket_cls.app, _.app)


if __name__ == '__main__':
    unittest.main()
