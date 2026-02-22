import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:lucide_icons/lucide_icons.dart';
import '../../providers/api_providers.dart';
import '../../api/models.dart' as model;
import 'package:intl/intl.dart';

class RecentActivityList extends ConsumerWidget {
  final bool shrinkWrap;
  const RecentActivityList({super.key, this.shrinkWrap = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final notificationsAsync = ref.watch(notificationsProvider);

    return notificationsAsync.when(
      data: (notifications) {
        if (notifications.isEmpty) {
          return const Center(child: Text("No recent classifications"));
        }

        return ListView.builder(
          shrinkWrap: shrinkWrap,
          physics: shrinkWrap ? const NeverScrollableScrollPhysics() : null,
          itemCount: notifications.length,
          itemBuilder: (context, index) {
            final notif = notifications[index];
            return _NotificationTile(notification: notif);
          },
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (err, stack) => Center(child: Text("Error: $err")),
    );
  }
}

class _NotificationTile extends ConsumerWidget {
  final model.Notification notification;

  const _NotificationTile({required this.notification});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final date = DateTime.tryParse(notification.timestamp) ?? DateTime.now();
    final formattedDate = DateFormat('MMM d, h:mm a').format(date);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 0, vertical: 8),
      child: ExpansionTile(
        leading: CircleAvatar(
          backgroundColor: theme.colorScheme.primaryContainer,
          child: Icon(
            _getIconForCategory(notification.predictedCategory),
            color: theme.colorScheme.onPrimaryContainer,
            size: 18,
          ),
        ),
        title: Text(
          notification.subject ?? "No Subject",
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: theme.colorScheme.secondaryContainer,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                notification.predictedCategory ?? "Unknown",
                style: TextStyle(
                  fontSize: 10,
                  color: theme.colorScheme.onSecondaryContainer,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              formattedDate,
              style: TextStyle(
                fontSize: 12,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _DetailRow(label: "From", value: notification.sender ?? ""),
                const SizedBox(height: 8),
                _DetailRow(label: "To", value: notification.recipient ?? ""),
                const SizedBox(height: 8),
                _DetailRow(
                  label: "Confidence",
                  value:
                      "${((notification.confidenceScore ?? 0) * 100).toStringAsFixed(1)}%",
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton.icon(
                      icon: const Icon(LucideIcons.check),
                      label: const Text("Correct"),
                      onPressed: () =>
                          _showCorrectionDialog(context, ref, notification),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  IconData _getIconForCategory(String? category) {
    switch (category?.toLowerCase()) {
      case 'focus':
        return LucideIcons.briefcase;
      case 'reference':
        return LucideIcons.bookOpen;
      case 'scheduling':
        return LucideIcons.calendar;
      case 'social':
        return LucideIcons.users;
      case 'ignore':
        return LucideIcons.trash2;
      default:
        return LucideIcons.mail;
    }
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  const _DetailRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 80,
          child: Text(
            "$label:",
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.grey,
            ),
          ),
        ),
        Expanded(child: Text(value)),
      ],
    );
  }
}

void _showCorrectionDialog(
  BuildContext context,
  WidgetRef ref,
  model.Notification notification,
) {
  showDialog(
    context: context,
    builder: (context) {
      return Consumer(
        builder: (context, ref, _) {
          final labelsAsync = ref.watch(labelsProvider);
          return AlertDialog(
            title: const Text("Correct Label"),
            content: labelsAsync.when(
              data: (labels) {
                return SizedBox(
                  width: double.maxFinite,
                  child: ListView(
                    shrinkWrap: true,
                    children: labels.map((label) {
                      // ignore: deprecated_member_use
                      return RadioListTile<String>(
                        title: Text(label),
                        value: label,
                        // ignore: deprecated_member_use
                        groupValue: notification.predictedCategory,
                        // ignore: deprecated_member_use
                        onChanged: (value) {
                          if (value != null) {
                            Navigator.of(context).pop();
                            _performCorrection(
                              context,
                              ref,
                              notification.id,
                              value,
                            );
                          }
                        },
                      );
                    }).toList(),
                  ),
                );
              },
              loading: () => const SizedBox(
                height: 100,
                child: Center(child: CircularProgressIndicator()),
              ),
              error: (e, s) => Text("Error loading labels: $e"),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text("Cancel"),
              ),
            ],
          );
        },
      );
    },
  );
}

void _performCorrection(
  BuildContext context,
  WidgetRef ref,
  String id,
  String newLabel,
) async {
  final client = ref.read(apiClientProvider);
  try {
    await client.correctLabel(id, newLabel);
    if (context.mounted) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Label corrected to $newLabel")));
      ref.invalidate(statsProvider);
      ref.invalidate(notificationsProvider);
    }
  } catch (e) {
    if (context.mounted) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Failed to correct label: $e")));
    }
  }
}
