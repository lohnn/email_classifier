import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/api_providers.dart';

class StatsChart extends ConsumerWidget {
  const StatsChart({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statsAsync = ref.watch(statsProvider);

    return statsAsync.when(
      data: (data) {
        if (data.stats.isEmpty) {
          return const Center(child: Text("No data available"));
        }

        final List<PieChartSectionData> sections = [];
        final keys = data.stats.keys.toList();
        final colors = [
          Colors.blueAccent,
          Colors.redAccent,
          Colors.greenAccent,
          Colors.orangeAccent,
          Colors.purpleAccent,
          Colors.tealAccent,
        ];

        for (int i = 0; i < keys.length; i++) {
          final key = keys[i];
          final value = data.stats[key]!;
          final color = colors[i % colors.length];

          sections.add(
            PieChartSectionData(
              color: color,
              value: value.toDouble(),
              title: '$value',
              radius: 60,
              titleStyle: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          );
        }

        return SizedBox(
          height: 300,
          child: Row(
            children: [
              Expanded(
                child: PieChart(
                  PieChartData(
                    sections: sections,
                    sectionsSpace: 2,
                    centerSpaceRadius: 40,
                  ),
                ),
              ),
              const SizedBox(width: 24),
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: keys.asMap().entries.map((entry) {
                  final index = entry.key;
                  final key = entry.value;
                  final color = colors[index % colors.length];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      children: [
                        Container(
                          width: 16,
                          height: 16,
                          decoration: BoxDecoration(
                            color: color,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          key,
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        );
      },
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (err, stack) => Center(child: Text("Error: $err")),
    );
  }
}
