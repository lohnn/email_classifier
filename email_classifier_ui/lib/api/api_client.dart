import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'models.dart';

class ApiClient {
  final Dio _dio;
  final String _apiKey;

  ApiClient()
    : _apiKey =
          dotenv.env['API_KEY'] ??
          (throw StateError(
            'API_KEY not found in .env file. '
            'Please add API_KEY=<your-key> to your .env file.',
          )),
      _dio = Dio(
        BaseOptions(
          baseUrl: dotenv.env['API_URL'] ?? 'http://127.0.0.1:8000',
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 30),
        ),
      );

  Future<StatsResponse> getStats({
    DateTime? startTime,
    DateTime? endTime,
  }) async {
    final queryParams = <String, dynamic>{};
    if (startTime != null) {
      queryParams['start_time'] = startTime.toIso8601String();
    }
    if (endTime != null) {
      queryParams['end_time'] = endTime.toIso8601String();
    }

    final response = await _dio.get('/stats', queryParameters: queryParams);
    return StatsResponse.fromJson(response.data);
  }

  Future<List<Notification>> getNotifications() async {
    final response = await _dio.get('/notifications');
    return (response.data as List)
        .map((e) => Notification.fromJson(e))
        .toList();
  }

  Future<void> ackNotifications(List<int>? ids) async {
    await _dio.post('/notifications/ack', data: {'ids': ids});
  }

  Future<List<Notification>> popNotifications() async {
    final response = await _dio.post('/notifications/pop');
    return (response.data as List)
        .map((e) => Notification.fromJson(e))
        .toList();
  }

  Future<List<String>> getLabels() async {
    final response = await _dio.get('/labels');
    return List<String>.from(response.data);
  }

  Future<RunResponse> runClassification({int limit = 20}) async {
    final response = await _dio.post('/run', queryParameters: {'limit': limit});
    return RunResponse.fromJson(response.data);
  }

  // /admin/force-check-corrections
  Future<void> forceCheckCorrections() async {
    await _dio.post(
      '/admin/force-check-corrections',
      options: Options(headers: {'X-API-Key': _apiKey}),
    );
  }

  Future<void> reclassify() async {
    await _dio.post(
      '/reclassify',
      options: Options(headers: {'X-API-Key': _apiKey}),
    );
  }

  Future<void> correctLabel(String logId, String correctedCategory) async {
    await _dio.post(
      '/logs/$logId/correction',
      data: {'corrected_category': correctedCategory},
      options: Options(headers: {'X-API-Key': _apiKey}),
    );
  }

  // Admin endpoints
  Future<void> triggerUpdate() async {
    await _dio.post(
      '/admin/trigger-update',
      options: Options(headers: {'X-API-Key': _apiKey}),
    );
  }

  Future<void> pushTrainingData() async {
    await _dio.post(
      '/admin/push-training-data',
      options: Options(headers: {'X-API-Key': _apiKey}),
    );
  }
}
