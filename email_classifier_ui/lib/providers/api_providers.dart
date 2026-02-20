import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../api/api_client.dart';
import '../api/models.dart';

final apiClientProvider = Provider<ApiClient>((ref) => ApiClient());

final statsProvider = FutureProvider.autoDispose<StatsResponse>((ref) async {
  final client = ref.watch(apiClientProvider);
  return client.getStats();
});

final notificationsProvider = FutureProvider.autoDispose<List<Notification>>((
  ref,
) async {
  final client = ref.watch(apiClientProvider);
  return client.getNotifications();
});

final labelsProvider = FutureProvider<List<String>>((ref) async {
  final client = ref.watch(apiClientProvider);
  return client.getLabels();
});
