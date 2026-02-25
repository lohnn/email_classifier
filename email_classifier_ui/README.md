# 📱 Email Classifier UI

A cross-platform companion app for the Email Classification System, built with **Flutter** and **Riverpod**.

## Features

- **Dashboard Overview** — Real-time statistics on email classifications by category.
- **Recent Activity Feed** — A stream of recently classified emails with confidence scores.
- **Manual Triggers** — Start classification jobs, trigger re-classification, or force a check for external corrections directly from the app.
- **Interactive Corrections** — Easily correct misclassified emails. Feedback is automatically sent to the brain to improve future models.

## Configuration

The app requires a `.env` file in the `email_classifier_ui/` directory:

```env
API_URL=http://<brain-ip>:8008
API_KEY=<your-admin-api-key>
```

- `API_URL`: The endpoint where the **brain** microservice is running.
- `API_KEY`: Must match the `ADMIN_API_KEY` configured in the brain's `.env`.

## Getting Started

1. **Install Flutter** — Follow the official [Flutter installation guide](https://docs.flutter.dev/get-started/install).
2. **Setup Environment** — Create the `.env` file as shown above.
3. **Install Dependencies**:
   ```bash
   flutter pub get
   ```
4. **Run the App**:
   ```bash
   flutter run
   ```

## Development

- **State Management**: [Riverpod](https://riverpod.dev/)
- **Networking**: [Dio](https://pub.dev/packages/dio)
- **Icons**: [Lucide Icons](https://lucideicons.com/)
