# Improved UI Flask App for Moral Machine

This application integrates the React-based frontend from `eva-redesign` with the Flask backend from `app`.

## Setup and Running

1.  **Backend Dependencies:**
    *   Navigate to this directory (`development/mortality/moral_machine/improved_ui_app/`).
    *   Create a Python virtual environment: `python -m venv .venv`
    *   Activate the virtual environment: `source .venv/bin/activate` (on Linux/macOS) or `.venv\Scripts\activate` (on Windows).
    *   Install Python dependencies: `pip install -r requirements.txt`
    *   Ensure you have a `.env` file in this directory with the necessary API keys (e.g., `ANTHROPIC_API_DEATH`, `OPENAI_API_DEATH`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`).

2.  **Frontend Build:**
    *   Navigate to the React app directory: `cd ../eva-redesign` (relative to this `improved_ui_app` folder).
    *   **Important:** Modify `eva-redesign/vite.config.ts` to ensure assets are served correctly when embedded in Flask. Change the `base` path for production builds:
        ```typescript
        // eva-redesign/vite.config.ts
        import { defineConfig } from "vite";
        import react from "@vitejs/plugin-react-swc";
        import path from "path";
        import { componentTagger } from "lovable-tagger";

        export default defineConfig(({ mode }) => ({
          base: mode === 'production' ? '/static/' : '/', // <-- Add this line or modify existing base value
          server: {
            host: "::",
            port: 8080,
          },
          plugins: [
            react(),
            mode === 'development' &&
            componentTagger(),
          ].filter(Boolean),
          resolve: {
            alias: {
              "@": path.resolve(__dirname, "./src"),
            },
          },
        }));
        ```
    *   Install Node.js dependencies (if not already done): `npm install` (or `yarn install`)
    *   Build the React application: `npm run build` (or `yarn build`)

3.  **Copy Frontend Assets:**
    *   Go back to the `improved_ui_app` directory.
    *   Create a `static` directory if it doesn't exist: `mkdir static` (though the Flask app creation might have implicitly done this if it ran and created an instance folder, it's good to be sure for the assets).
    *   Copy the *contents* of `../eva-redesign/dist/` into the `./static/` directory.
        *   For example, `../eva-redesign/dist/index.html` should become `./static/index.html`.
        *   All assets (like JS, CSS files, often in an `assets` subfolder within `dist`) should be copied into `./static/` (e.g. `./static/assets/...`).

4.  **Run the Application:**
    *   Make sure you are in the `improved_ui_app` directory and your Python virtual environment is activated.
    *   Run the Flask application: `python flask_app.py`
    *   Open your browser and navigate to `http://127.0.0.1:5000` (or the port specified in `flask_app.py`).

## Project Structure (`improved_ui_app`)

*   `flask_app.py`: The main Flask application file. Serves the React frontend and provides API endpoints.
*   `requirements.txt`: Python dependencies.
*   `static/`: This is where the built React frontend assets (from `eva-redesign/dist/`) must be copied.
*   `templates/`: (Currently likely unused if all UI is from React) For any Flask-rendered HTML templates.
*   `.env`: (You need to create this) For API keys and other environment variables.
*   `.venv/`: Python virtual environment (after you create it).
*   `instance/cache/`: Directory for caching LLM responses (created automatically by the Flask app).

## Development Workflow (Alternative)

For a more interactive development experience with hot-reloading for the frontend:

1.  Run the Vite dev server for the React app:
    *   In `eva-redesign/`, run `npm run dev` (usually serves on `http://localhost:8080`).
2.  Run the Flask dev server:
    *   In `improved_ui_app/`, run `python flask_app.py` (usually serves on `http://localhost:5000`).
3.  Configure CORS in `improved_ui_app/flask_app.py` to allow requests from the Vite dev server origin (e.g., `http://localhost:8080`). You can use the `flask-cors` extension for this.
    ```python
    # In flask_app.py
    from flask_cors import CORS
    # ...
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:8080"}}) # Example for API routes
    # ...
    ```
    *   Alternatively, use Vite's proxy feature in `eva-redesign/vite.config.ts` to proxy API requests from the React app to the Flask backend. See Vite documentation for `server.proxy`.
4.  Access the React app via the Vite dev server URL (e.g., `http://localhost:8080`). API calls will be directed to your Flask backend.

This separate server setup is only for development. For deployment, follow the main setup steps to serve the React build files directly from Flask. 