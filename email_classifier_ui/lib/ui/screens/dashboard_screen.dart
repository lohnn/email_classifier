import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../widgets/stats_chart.dart';
import '../widgets/recent_activity_list.dart';
import '../../providers/api_providers.dart';
import 'package:lucide_icons/lucide_icons.dart';

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Email Classifier'),
        actions: [
          IconButton(
            icon: const Icon(LucideIcons.bean),
            tooltip: 'Reclassify emails',
            onPressed: () async {
              // Trigger classification
              final client = ref.read(apiClientProvider);
              try {
                await client.runReClassification();
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        "Reclassified emails",
                      ),
                    ),
                  );
                  ref.invalidate(statsProvider);
                  ref.invalidate(notificationsProvider);
                }
              } catch (e, stackTrace) {
                debugPrint(e.toString());
                debugPrintStack(stackTrace: stackTrace);
                if (context.mounted) {
                  ScaffoldMessenger.of(
                    context,
                  ).showSnackBar(SnackBar(content: Text("Error: $e")));
                }
              }
            },
          ),
          IconButton(
            icon: const Icon(LucideIcons.refreshCw),
            tooltip: 'Refresh data',
            onPressed: () {
              ref.invalidate(statsProvider);
              ref.invalidate(notificationsProvider);
            },
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth > 800) {
            // Desktop / Wide Layout
            return Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 4,
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Overview",
                          style: Theme.of(context).textTheme.headlineMedium,
                        ),
                        const SizedBox(height: 24),
                        const StatsChart(),
                      ],
                    ),
                  ),
                ),
                Expanded(
                  flex: 6,
                  child: Container(
                    decoration: BoxDecoration(
                      border: Border(
                        left: BorderSide(
                          color: Theme.of(
                            context,
                          ).dividerColor.withValues(alpha: 0.1),
                        ),
                      ),
                    ),
                    child: const RecentActivityList(),
                  ),
                ),
              ],
            );
          } else {
            // Mobile Layout
            return SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Overview",
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 16),
                  const SizedBox(height: 300, child: StatsChart()),
                  const SizedBox(height: 24),
                  Text(
                    "Recent Activity",
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 16),
                  const RecentActivityList(shrinkWrap: true),
                ],
              ),
            );
          }
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () async {
          // Trigger classification
          final client = ref.read(apiClientProvider);
          try {
            final result = await client.runClassification();
            if (context.mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text("Processed ${result.processedCount} emails"),
                ),
              );
              ref.invalidate(statsProvider);
              ref.invalidate(notificationsProvider);
            }
          } catch (e) {
            if (context.mounted) {
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(SnackBar(content: Text("Error: $e")));
            }
          }
        },
        label: const Text("Run Now"),
        icon: const Icon(LucideIcons.play),
      ),
    );
  }
}
