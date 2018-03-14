import asyncio
import unittest.mock

import mock

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
        with unittest.mock.patch(
                'oracle.oracle_server.update_omega', mock_update_omega
             ), unittest.mock.patch(
                'oracle.oracle_server.schedule_task', mock_schedule_task
             ):
                module_ut.main(_.port, 'test_temp')
        _.aiohttp_web.Application.assert_called_once_with()
        self.assertEqual(_.app.on_startup, [_.partial_train, _.scheduled_job])
        self.assertEqual(_.app.on_shutdown, [_.websocket_shutdown])
        mock_schedule_task.assert_called_once_with(_.update_job,
            module_ut.datetime.timedelta(seconds=5))
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
        _.time.sleep.assert_called_once_with(3)
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
            unittest.mock.call(.1),
            unittest.mock.call(.1),
            unittest.mock.call(.1),
        ], any_order=True)
        self.assertEqual(mock_sleep.call_count, 3)
        _.est_Ω.train_ensemble.assert_has_calls([
            unittest.mock.call('faketempdir/oracle_keras_0.h5'),
            unittest.mock.call('faketempdir/oracle_keras_1.h5'),
            unittest.mock.call('faketempdir/oracle_keras_2.h5'),
        ], any_order=True)
        self.assertEqual(_.est_Ω.train_ensemble.call_count, 3)

    @unittest.mock.patch('oracle.oracle_server.est_Ω', _.est_Ω)
    @unittest.mock.patch('oracle.oracle_server.WebSocket', _.websocket_cls)
    def test_update_omega(self):
        prediction_df = module_ut.pandas.DataFrame(
            {'Ω' : [1, 2, 3, 2, 4], 'Ω_est': [4, 2, 3, 1, 3]},
            index=module_ut.pandas.date_range(
                start=module_ut.datetime.datetime(1987, 7, 17), periods=5
            )
        )
        mock_predict_live = mymock(prediction_df)
        _.loaded_model.predict_live = mock_predict_live
        mock_load_model = mymock(_.loaded_model)
        _.websocket_cls.put_message = mymock(None)
        module_ut.update_omega(mock_load_model)()
        _.websocket_cls.put_message.assert_called_once_with(
          '[{"omega": 1, "omega_est": 4, "time": "1987-07-17T00:00:00.000Z"},'
          ' {"omega": 2, "omega_est": 2, "time": "1987-07-18T00:00:00.000Z"},'
          ' {"omega": 3, "omega_est": 3, "time": "1987-07-19T00:00:00.000Z"},'
          ' {"omega": 2, "omega_est": 1, "time": "1987-07-20T00:00:00.000Z"},'
          ' {"omega": 4, "omega_est": 3, "time": "1987-07-21T00:00:00.000Z"}]'
        )

    def test_index(self):
        _.request.match_info = {}
        mock_read_file = mymock(_.file_contents)
        mock_listdir = mymock(['anything except index.html'])
        with unittest.mock.patch(
                'oracle.oracle_server._read_file', mock_read_file
            ), unittest.mock.patch(
                'oracle.oracle_server.os.listdir', mock_listdir
            ), self.assertRaises(ValueError):
                asyncio.get_event_loop().run_until_complete(
                   module_ut.index('myfakedir')(_.request)
                )
        mock_listdir.assert_called_once_with('myfakedir')
        mock_read_file.assert_not_called()
        mock_response_cls = mymock(_.ret)
        mock_listdir = mymock(['somefile', 'index.html', 'different file'])
        with unittest.mock.patch(
                'oracle.oracle_server._read_file', mock_read_file
            ), unittest.mock.patch(
                'oracle.oracle_server.os.listdir', mock_listdir
            ), unittest.mock.patch(
                'oracle.oracle_server.aiohttp.web.Response', mock_response_cls
            ):
                ret = asyncio.get_event_loop().run_until_complete(
                   module_ut.index('myfakedir')(_.request)
                )
        mock_response_cls.assert_called_once_with(body=_.file_contents,
                content_type='text/html')
        mock_listdir.assert_called_once_with('myfakedir')
        mock_read_file.assert_called_once_with('myfakedir', 'index.html')
        self.assertEqual(ret, _.ret)

        _.request.match_info = {'page': '/style.css'}
        mock_listdir = mymock(['style.css', 'index.html', 'different file'])
        with unittest.mock.patch(
                'oracle.oracle_server._read_file', mymock(_.file_contents)
            ) as mock_read_file, unittest.mock.patch(
                'oracle.oracle_server.os.listdir', mock_listdir
            ), unittest.mock.patch(
                'oracle.oracle_server.aiohttp.web.Response', mymock(_.ret)
            ) as mock_response_cls:
                ret = asyncio.get_event_loop().run_until_complete(
                   module_ut.index('other/dir')(_.request)
                )
        mock_response_cls.assert_called_once_with(body=_.file_contents,
                content_type='text/css')
        mock_listdir.assert_called_once_with('other/dir')
        mock_read_file.assert_called_once_with('other/dir', 'style.css')
        self.assertEqual(ret, _.ret)

        _.request.match_info = {'page': 'logo.jpg'}
        mock_listdir = mymock(['style.css', 'index.html', 'logo.jpg'])
        with unittest.mock.patch(
                'oracle.oracle_server._read_file', mymock(_.file_contents)
            ) as mock_read_file, unittest.mock.patch(
                'oracle.oracle_server.os.listdir', mock_listdir
            ), unittest.mock.patch(
                'oracle.oracle_server.aiohttp.web.Response', mymock(_.ret)
            ) as mock_response_cls:
                ret = asyncio.get_event_loop().run_until_complete(
                   module_ut.index('other/dir')(_.request)
                )
        mock_response_cls.assert_called_once_with(body=_.file_contents)
        mock_listdir.assert_called_once_with('other/dir')
        mock_read_file.assert_called_once_with('other/dir', 'logo.jpg')
        self.assertEqual(ret, _.ret)

    def test_read_file(self):
        with mock.patch(
                'builtins.open', mock.mock_open(read_data='my_data')
             ) as m_open:
            ret = module_ut._read_file('my/dir', 'page.html')
        self.assertEqual(ret, 'my_data')
        m_open.assert_called_once_with('my/dir/page.html', 'rb')


if __name__ == '__main__':
    unittest.main()
