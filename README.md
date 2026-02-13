# 📧 Email Classifier

A smart email classification system with a cross-platform companion app.

This monorepo contains two projects:

| Directory                                            | Description                                                                                                                                                                  | Stack   |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| [`email_classifier_brain/`](email_classifier_brain/) | Few-shot email classifier powered by **SetFit** and the **intfloat/multilingual-e5-small** embedding model. Trains on a workstation, runs inference on a **Raspberry Pi 4**. | Python  |
| [`email_classifier_ui/`](email_classifier_ui/)       | Cross-platform email reader for browsing classified emails, with support for correcting misclassifications.                                                                  | Flutter |

## Overview

The **brain** handles all the ML heavy-lifting — training on labelled email
examples and classifying incoming mail into categories (e.g. `URGENT`, `FOCUS`,
`REFERENCE`, `NOISE`). It uses rich metadata like sender, role, mass-mail
detection, and attachment types to make accurate predictions.

The **UI** is a Flutter email reader designed for browsing emails organised by
their classification. If the model gets it wrong, users can correct the category
directly from the app — feeding improvements back into the training data.

## Getting Started

Each subproject has its own README with setup instructions:

- **Brain** — see
  [`email_classifier_brain/README.md`](email_classifier_brain/README.md) for
  training data format, configuration, and deployment.
- **UI** — see [`email_classifier_ui/README.md`](email_classifier_ui/README.md)
  for Flutter setup and development.

## License

This project is open source. See the individual subproject directories for
details.
