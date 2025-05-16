# AI Mortality Experiment - Flask Backend

This project is the Flask-based backend API for the "AI Mortality Experiment" React frontend. It handles scenario processing, interaction with various Large Language Models (LLMs), data caching, and serves API endpoints consumed by the frontend.

## Prerequisites

*   Python (3.8+ recommended)
*   `pip` (Python package installer)
*   A way to create Python virtual environments (e.g., `venv` module)

## Setup

1.  **Clone the Repository:**
    ```bash
    # Replace with your actual repository URL if different
    git clone <YOUR_FLASK_BACKEND_GIT_REPOSITORY_URL>
    cd mortality_flask 
    ```

2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python -m venv .venv
    ```
    On Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```
    On Windows:
    ```bash
    .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Ensure your virtual environment is activated.
    ```bash
    pip install -r requirements.txt
    ```
    The application uses NLTK for text analysis. Necessary NLTK data packages (stopwords, punkt, wordnet, omw-1.4) will be downloaded automatically on the first run of the application if not found, which requires an internet connection.

4.  **Configure Environment Variables:**
    Create a `.env` file in the `mortality_flask` root directory. This file stores sensitive API keys. Add the following keys with your actual credentials:
    ```env
    ANTHROPIC_API_DEATH="your_anthropic_api_key"
    OPENAI_API_DEATH="your_openai_api_key"
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    ```
    The application will attempt to load these variables.

## Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Start the Flask Development Server:**
    ```bash
    python flask_app.py
    ```
    The API server will typically start on `http://127.0.0.1:5000`.

## Key API Endpoints

The backend exposes several API endpoints, including:

*   `GET /`: Health check for the API.
*   `GET /api/providers`: Returns a list of available LLM providers.
*   `POST /api/scenario/initiate_processing`: Initiates the processing of a scenario. Takes scenario data and provider details, returns initial reasoning from the LLM.
*   `POST /api/scenario/get_decision`: Based on a scenario hash (from initiation), prompts the LLM for a final decision.
*   `POST /api/scenario/finalize_and_get_result`: Takes a scenario hash, performs final analysis (word frequency, philosophical alignment), and returns the complete processed scenario result.
*   `GET /api/get-scenario-result/<scenario_hash>`: Retrieves a fully processed and cached scenario result by its hash.
*   `GET /api/alignment-report-data`: Provides data for generating an alignment report, sourced from a CSV file (`instance/v4_quant_result.csv`).

Refer to `flask_app.py` for detailed request/response structures.

## Caching

The application uses a local caching system to store scenario results and intermediate data to avoid redundant LLM calls and processing:

*   **Scenario Cache**: `instance/all_scenario_cache.json` stores detailed results of processed scenarios, keyed by a unique scenario hash.
*   **Philosophy Cache**: `instance/philosophy_cache.json` caches philosophical alignment classifications for reasoning texts.
*   **Quantitative Results**: `instance/v4_quant_result.csv` is used by the `/api/alignment-report-data` endpoint.

The `instance/` folder is created automatically in the `mortality_flask` directory if it doesn't exist.

## Serving the Frontend (Production / Unified Setup)

While the Flask backend can run independently (see Development Workflow), it's also configured to serve a built version of the `mortality_react` frontend.

1.  **Build the React Frontend:** Navigate to your `mortality_react` project directory and run its build command (e.g., `npm run build`).
2.  **Copy Assets:** Copy the contents of the `mortality_react/dist/` directory into the `mortality_flask/static/frontend_dist/` directory. The Flask app will then serve the `index.html` from `static/frontend_dist/` for root and client-side routes, and other static assets (JS, CSS) from the same location under the `/static/` URL path (e.g., `/static/assets/main.js` will serve `mortality_flask/static/frontend_dist/assets/main.js`).

The `flask_app.py` includes routes to handle serving these static files.

## Development Workflow

For a typical development setup:

1.  **Run the Flask Backend:**
    Start the Flask development server as described above (usually on `http://127.0.0.1:5000`).
2.  **Run the React Frontend:**
    In your separate `mortality_react` project directory, start its development server (e.g., `npm run dev`, typically on `http://localhost:8080`).
3.  **CORS:**
    The Flask application (`flask_app.py`) is configured with `Flask-CORS` to allow cross-origin requests from the React development server (e.g. `http://localhost:8080`).

This setup allows for independent development and hot-reloading for both the frontend and backend.

## Technologies Used

*   **Python**
*   **Flask**: Micro web framework for building the API.
*   **Flask-CORS**: For handling Cross-Origin Resource Sharing.
*   **NLTK**: Natural Language Toolkit for text processing (word frequency analysis).
*   **python-dotenv**: For managing environment variables.
*   APIs for various LLMs (Anthropic, OpenAI, DeepSeek, Gemini).

## Contact

For questions, suggestions, or collaborations regarding the "AI Mortality Experiment" project, please reach out to:

-   **Victor Löfgren @ Cloudwalk**: [victor.sattamini@cloudwalk.io]

---
