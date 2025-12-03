# Portfolio Website Template

This repository contains a Django-based portfolio website template, designed to be deployed on Google Cloud Platform (GCP) Cloud Run.

## Project Structure

- `portfolio_core/`: The main Django project configuration.
- `main_portfolio/`: The Django app for the portfolio content.
- `Dockerfile`: Configuration for building the Docker image.
- `docker-compose.yml`: Configuration for local testing with Docker Compose.
- `.github/workflows/ci.yml`: CI/CD workflow for testing and deployment.

## Local Development

You can run the project locally using Docker Compose or directly with Python/uv.

### Using Docker Compose (Recommended)

1.  **Build and run the container:**
    ```bash
    docker-compose up --build
    ```
2.  **Access the website:**
    Open your browser and navigate to `http://localhost:8000`.

### Using uv (Local Python)

1.  **Install dependencies:**
    ```bash
    uv sync
    ```
2.  **Apply migrations:**
    ```bash
    uv run python manage.py migrate
    ```
3.  **Run the development server:**
    ```bash
    uv run python manage.py runserver
    ```

## Deployment to GCP Cloud Run

This repository is set up for Continuous Deployment to GCP Cloud Run via GitHub Actions.

### Prerequisites

1.  **GCP Project:** Ensure you have a Google Cloud Platform project.
2.  **Cloud Run API:** Enable the Cloud Run API.
3.  **Service Account:** Create a service account with permissions to deploy to Cloud Run (e.g., `roles/run.admin`, `roles/iam.serviceAccountUser`).
4.  **Workload Identity Federation:** Set up Workload Identity Federation to allow GitHub Actions to authenticate as your service account.

### Configuration

Update the `.github/workflows/ci.yml` file with your GCP details:
- Uncomment the deployment steps.
- Set `workload_identity_provider` and `service_account`.
- Update the `service` name and `region`.

### Manual Deployment

You can also deploy manually using the `gcloud` CLI:

```bash
gcloud run deploy my-portfolio \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Customization

-   **Templates:** Modify the templates in `main_portfolio/templates/` (you may need to create this directory).
-   **Static Files:** Add your CSS, JS, and images to `main_portfolio/static/`.
-   **Views:** Update `main_portfolio/views.py` to change the logic.

## "Clean Up" Note

This repository was refactored from a previous tool ("PapAIrus"). All unrelated files have been removed to provide a clean slate for your portfolio.
