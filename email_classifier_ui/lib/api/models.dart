class Notification {
  final String id;
  final String timestamp;
  final String? sender;
  final String? recipient;
  final String? subject;
  final String? predictedCategory;
  final double? confidenceScore;
  final bool isRead;

  Notification({
    required this.id,
    required this.timestamp,
    this.sender,
    this.recipient,
    this.subject,
    this.predictedCategory,
    this.confidenceScore,
    required this.isRead,
  });

  factory Notification.fromJson(Map<String, dynamic> json) {
    return Notification(
      id: json['id'],
      timestamp: json['timestamp'],
      sender: json['sender'],
      recipient: json['recipient'],
      subject: json['subject'],
      predictedCategory: json['predicted_category'],
      confidenceScore: json['confidence_score']?.toDouble(),
      isRead: json['is_read'] == true || json['is_read'] == 1,
    );
  }
}

class StatsResponse {
  final Map<String, int> stats;

  StatsResponse({required this.stats});

  factory StatsResponse.fromJson(Map<String, dynamic> json) {
    return StatsResponse(stats: Map<String, int>.from(json['stats']));
  }
}

class RunResponse {
  final String status;
  final int processedCount;
  final List<dynamic> details;

  RunResponse({
    required this.status,
    required this.processedCount,
    required this.details,
  });

  factory RunResponse.fromJson(Map<String, dynamic> json) {
    return RunResponse(
      status: json['status'],
      processedCount: json['processed_count'],
      details: json['details'] ?? [],
    );
  }
}

class CorrectionRequest {
  final String correctedCategory;

  CorrectionRequest({required this.correctedCategory});

  Map<String, dynamic> toJson() {
    return {'corrected_category': correctedCategory};
  }
}

class AckRequest {
  final List<int>? ids;

  AckRequest({this.ids});

  Map<String, dynamic> toJson() {
    return {'ids': ids};
  }
}
