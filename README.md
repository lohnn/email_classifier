# 📧 Email Classifier

A smart email classification system with a cross-platform companion app.

This monorepo contains two projects:

| Directory                                            | Description                                                                                                                                                                  | Stack            |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| [`email_classifier_brain/`](email_classifier_brain/) | Few-shot email classifier and FastAPI microservice. Powered by **SetFit** and **intfloat/multilingual-e5-small**. Runs background jobs for auto-classification and feedback. | Python, FastAPI  |
| [`email_classifier_ui/`](email_classifier_ui/)       | Cross-platform companion app for browsing classified emails, viewing statistics, and correcting misclassifications.                                                           | Flutter, Riverpod |

## Overview

The **brain** handles all the ML heavy-lifting — training on labelled email
examples and classifying incoming mail into categories (e.g. `URGENT`, `FOCUS`,
`REFERENCE`, `NOISE`). It runs as a microservice on a **Raspberry Pi 4**,
automatically processing new mail via IMAP and exposing an API for the UI.

The **UI** is a Flutter-based dashboard designed for monitoring the system and
browsing organised emails. If the model gets it wrong, users can correct the
category directly from the app — which automatically appends the email to the
training data and triggers a future retrain.

## Getting Started

Each subproject has its own README with detailed setup instructions:

- **Brain** — see
  [`email_classifier_brain/README.md`](email_classifier_brain/README.md) for
  API documentation, background jobs, and deployment.
- **UI** — see [`email_classifier_ui/README.md`](email_classifier_ui/README.md)
  for Flutter environment setup and configuration.

## License

This project is open source. See the individual subproject directories for
details.
