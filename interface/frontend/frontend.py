import os
import time
import pandas as pd
import io
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    flash,
    get_flashed_messages,
    jsonify,
)
from werkzeug.utils import secure_filename
from jinja2 import DictLoader, Environment


# --- Configuration and Initialization ---
UPLOAD_FOLDER = "uploads"
ALLOWED_UPLOAD_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ⚠️ INFLUENCER DATA SOURCE (CSV) ⚠️
DEFAULT_CSV_PATH = r"C:\Users\Adithya\Dev\newdataset\top100_with_sentiment.csv"

# --- Global State Simulation ---
APP_STATE = {
    "influencer_data": None,  # Stores the list of influencer dictionaries
    "data_loaded": False,
}

# --- Mock CSV Data to be returned by the simulated backend API ---
MOCK_API_CSV_RESPONSE = """
rank,influencer,category,followers,likes,comments,sentiment_score
1,ai_superstar,fashion,500000,80000,12000,0.95
2,future_foodie,food,120000,30000,4500,0.88
3,pet_bot_42,pet,85000,21000,3100,0.72
4,travel_ai_guide,travel,90000,25000,3500,0.65
5,home_design_ai,interior,60000,15000,2000,0.40
"""

# --- Categories for the filter page ---
CATEGORIES_LIST = [
    ("C1", "fashion", "Trend analysis and clothing haul videos."),
    ("C2", "travel", "Vlogs from unique destinations and travel tips."),
    ("C3", "family", "Parenting, kids, and general family lifestyle content."),
    ("C4", "food", "Recipes, restaurant reviews, and culinary arts."),
    ("C5", "fitness", "Fitness routines, nutrition, and workout tips."),
    ("C6", "beauty", "Makeup, skincare, and cosmetic reviews."),
    ("C7", "pet", "Animal care, training, and pet lifestyle."),
    ("C8", "interior", "Home decor, renovation projects, and interior design."),
    ("C9", "other", "Miscellaneous or uncategorized content."),
]


# --- Utility Functions ---


def allowed_file(filename):
    """Check if the uploaded file has the allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS
    )


def format_number(n):
    """Simple helper to format large numbers for display."""
    if not isinstance(n, (int, float)):
        return str(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n) if isinstance(n, float) and n.is_integer() else n)


def get_sentiment_emoji(score):
    """Converts a numeric score to a corresponding sentiment emoji."""
    try:
        score = float(score)
        if score > 0.8:
            return "🤩"
        elif score > 0.3:
            return "😊"
        else:
            return "😞"
    except (ValueError, TypeError):
        return "❓"


def load_data_from_csv_content(csv_content):
    """Loads influencer data from a CSV string content (e.g., from API response)."""
    try:
        # Use StringIO to treat the string as a file
        df = pd.read_csv(io.StringIO(csv_content))

        df.columns = df.columns.str.lower().str.strip()

        # Standardize column names
        if "sentiment_sentiment_" in df.columns:
            df.rename(columns={"sentiment_sentiment_": "sentiment_score"}, inplace=True)
        elif "sentiment_" in df.columns:
            df.rename(columns={"sentiment_": "sentiment_score"}, inplace=True)

        if "category" not in df.columns:
            raise ValueError("CSV must contain a 'category' column for filtering.")

        APP_STATE["influencer_data"] = df.to_dict("records")
        APP_STATE["data_loaded"] = True
        return True, "Data successfully loaded from API response."

    except Exception as e:
        return False, f"Error processing API CSV content: {e}"


def load_data_from_path(filepath):
    """Loads CSV/XLSX influencer data from a local file path."""
    absolute_filepath = os.path.abspath(filepath)
    print(f"DEBUG: Attempting to load data from absolute path: {absolute_filepath}")

    try:
        if not os.path.exists(absolute_filepath):
            raise FileNotFoundError(f"File not found at: {absolute_filepath}")

        file_ext = absolute_filepath.rsplit(".", 1)[1].lower()

        if file_ext == "csv":
            df = pd.read_csv(absolute_filepath)
        elif file_ext in ["xlsx", "xls"]:
            df = pd.read_excel(absolute_filepath)
        else:
            raise ValueError(
                "Unsupported file format for INFLUENCER DATA. Must be CSV or XLSX."
            )

        df.columns = df.columns.str.lower().str.strip()

        if "sentiment_sentiment_" in df.columns:
            df.rename(columns={"sentiment_sentiment_": "sentiment_score"}, inplace=True)
        elif "sentiment_" in df.columns:
            df.rename(columns={"sentiment_": "sentiment_score"}, inplace=True)

        if "category" not in df.columns:
            raise ValueError("CSV must contain a 'category' column for filtering.")

        APP_STATE["influencer_data"] = df.to_dict("records")
        APP_STATE["data_loaded"] = True
        return True, "Influencer data successfully loaded from local path."

    except FileNotFoundError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error loading influencer data: {e}"


# --- HTML TEMPLATES (Jinja Syntax) ---

TEMPLATES = {
    "base": """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Data Driven Influencer Marketing{% endblock title %}</title>
    <!-- Tailwind CSS CDN for quick and modern styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .nav-link:hover { background-color: rgba(255, 255, 255, 0.1); }
        .card { transition: transform 0.2s; }
        .card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); }
        /* Simple loader animation */
        .loader {
            border: 4px solid #f3f3f3; /* Light grey */
            border-top: 4px solid #4f46e5; /* Indigo */
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <nav class="bg-indigo-700 p-4 shadow-lg sticky top-0 z-10">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <a href="{{ url_for('landing') }}" class="text-2xl font-bold text-white tracking-wider">
                Data Driven Influencer Marketing
            </a>
            <div class="space-x-4 flex">
                <a href="{{ url_for('landing') }}" class="nav-link text-white px-3 py-2 rounded-lg text-sm font-medium">Home</a>
                <a href="{{ url_for('upload_data') }}" class="nav-link text-white px-3 py-2 rounded-lg text-sm font-medium">1. Data</a>
                <a href="{{ url_for('categories') }}" class="nav-link text-white px-3 py-2 rounded-lg text-sm font-medium">2. Categories</a>
                <a href="{{ url_for('influencers') }}" class="nav-link text-white px-3 py-2 rounded-lg text-sm font-medium">3. Results</a>
            </div>
        </div>
    </nav>

    <div class="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="mb-4" id="flash-messages">
                {% for category, message in messages %}
                    <div class="p-3 rounded-lg text-sm {% if category == 'success' %}bg-green-100 text-green-800{% elif category == 'error' %}bg-red-100 text-red-800{% else %}bg-yellow-100 text-yellow-800{% endif %}" role="alert">
                        {{ message }}
                    </div>
                {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <div class="bg-white p-8 rounded-xl shadow-2xl">
            {% block content %}{% endblock content %}
        </div>
    </div>

    <footer class="mt-12 p-4 bg-gray-200 text-center text-gray-600 text-sm">
        &copy; 2025 Data Driven Influencer Marketing. Powered by Flask.
    </footer>
</body>
</html>
""",
    "landing": """
{% extends "base" %}
{% block title %}Welcome to Data Driven Influencer Marketing{% endblock title %}
{% block content %}
    <header class="text-center pb-8">
        <h1 class="text-5xl font-extrabold text-gray-900 mb-4">
            Data Driven Influencer Matching
        </h1>
        <p class="text-xl text-gray-600 max-w-2xl mx-auto">
            Find the perfect voices for your brand using predictive analytics and deep audience insights.
        </p>
    </header>

    <div class="grid md:grid-cols-3 gap-8 text-center mt-10">
        <div class="p-6 bg-indigo-50 rounded-xl card border border-indigo-200">
            <div class="text-indigo-600 mb-3 text-4xl">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 mx-auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 0 0-9.8 11.2 5 5 0 0 1 7 5.8H15a5 5 0 0 1 7-5.8A10 10 0 0 0 12 2z"/><circle cx="12" cy="7" r="2"/><path d="M16 16.5c-1.3 1.2-3.1 1.7-5 1.7-1.9 0-3.7-.5-5-1.7"/></svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-800">1. Upload Text Data</h3>
            <p class="text-gray-500 mt-2">Upload a TXT file containing additional user data for analysis.</p>
            <a href="{{ url_for('upload_data') }}" class="mt-4 inline-block text-indigo-600 font-medium hover:text-indigo-800">
                Go to Upload &rarr;
            </a>
        </div>
        <div class="p-6 bg-indigo-50 rounded-xl card border border-indigo-200">
            <div class="text-indigo-600 mb-3 text-4xl">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 mx-auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-800">2. Define Target Niche</h3>
            <p class="text-gray-500 mt-2">Select the categories that best match your campaign goals.</p>
            <a href="{{ url_for('categories') }}" class="mt-4 inline-block text-indigo-600 font-medium hover:text-indigo-800">
                Choose Categories &rarr;
            </a>
        </div>
        <div class="p-6 bg-indigo-50 rounded-xl card border border-indigo-200">
            <div class="text-indigo-600 mb-3 text-4xl">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 mx-auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 19c-1.5 0-3-.5-4-1.5s-2.5-1.5-4-1.5-3 .5-4 1.5l-1 1V4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v15l-1-1z"/><circle cx="12" cy="7" r="4"/></svg>
            </div>
            <h3 class="text-lg font-semibold text-gray-800">3. View Matched Talent</h3>
            <p class="text-gray-500 mt-2">See a list of top-ranking influencers for your specific needs.</p>
            <a href="{{ url_for('influencers') }}" class="mt-4 inline-block text-indigo-600 font-medium hover:text-indigo-800">
                View Results &rarr;
            </a>
        </div>
    </div>
{% endblock content %}
""",
    "upload": """
{% extends "base" %}
{% block title %}Upload Dataset{% endblock title %}
{% block content %}
    <h2 class="text-3xl font-bold text-gray-900 mb-6 border-b pb-2">Step 1: Upload User Text Data</h2>
    <p class="text-gray-600 mb-4">
        The main influencer dataset is loaded automatically from the CSV path defined in 
        <code class="bg-gray-200 p-1 rounded-md text-sm">app.py</code> (<code>{{ DEFAULT_CSV_PATH }}</code>).
    </p>

    <div id="data-status-message">
        {% if APP_STATE['data_loaded'] %}
            <p class="mt-4 text-sm text-green-600 font-medium p-3 bg-green-50 rounded-lg border border-green-200">
                ✅ Influencer data loaded successfully! Proceed below to upload custom text data or go to categories.
            </p>
        {% else %}
            <p class="mt-4 text-sm text-red-500 font-medium p-3 bg-red-50 rounded-lg border border-red-200">
                ⚠️ Influencer data not loaded. Check the CSV path in <code>app.py</code> and the file existence.
            </p>
        {% endif %}
    </div>


    <p class="text-gray-600 mt-6 mb-8 border-t pt-4">Upload a custom <strong>.txt</strong> file for backend processing (simulated):</p>

    <form id="upload-form" class="max-w-xl mx-auto p-6 bg-indigo-50 rounded-xl shadow-lg border border-indigo-200">
        <label for="file-upload" class="block text-lg font-medium text-gray-700 mb-3">Select Custom Text File (.txt)</label>

        <div class="flex items-center space-x-4">
            <input id="file-upload" name="file" type="file" accept=".txt" required class="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-indigo-500 file:text-white
                hover:file:bg-indigo-600
                cursor-pointer
            ">
            <button type="submit" id="upload-button" class="inline-flex justify-center py-2 px-6 border border-transparent shadow-sm text-sm font-medium rounded-full text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition duration-150">
                Upload Text
            </button>
            <div id="loader-spinner" class="hidden loader"></div>
        </div>
        <p class="mt-2 text-xs text-gray-500">Supported format: .txt</p>
    </form>

    <script>
        // --- ASYNCHRONOUS UPLOAD AND DATA REPLACEMENT LOGIC ---
        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();

            const form = e.target;
            const fileInput = document.getElementById('file-upload');
            const button = document.getElementById('upload-button');
            const spinner = document.getElementById('loader-spinner');
            const file = fileInput.files[0];
            const flashMessages = document.getElementById('flash-messages') || document.createElement('div');
            flashMessages.id = 'flash-messages';

            // 1. Validation and Setup
            if (!file) {
                alert('Please select a .txt file to upload.');
                return;
            }

            const allowedExt = ['.txt'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedExt.includes(fileExt)) {
                 // Clear previous messages and show error
                flashMessages.innerHTML = `
                    <div class="p-3 rounded-lg text-sm bg-red-100 text-red-800" role="alert">
                        Invalid file type. Only .txt files are allowed.
                    </div>
                `;
                if (!document.getElementById('flash-messages')) {
                    document.querySelector('.max-w-7xl.mx-auto.p-4').prepend(flashMessages);
                }
                return;
            }

            const formData = new FormData(form);

            // 2. Show Loading State
            button.classList.add('hidden');
            spinner.classList.remove('hidden');
            form.querySelector('input[type="file"]').disabled = true;

            try {
                // Clear previous flash messages
                flashMessages.innerHTML = '';

                // 3. Send file to the simulated backend API
                const response = await fetch("{{ url_for('process_txt_api') }}", {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                // 4. Handle Response and Data Replacement
                if (response.ok && result.success) {
                    // Success: Load the new CSV data returned by the backend

                    // --- Display Success Message ---
                    const successMsg = `
                        <div class="p-3 rounded-lg text-sm bg-green-100 text-green-800" role="alert">
                            ✅ Text file uploaded successfully and backend processing complete! New influencer data loaded.
                        </div>
                    `;
                    flashMessages.innerHTML = successMsg;
                    if (!document.getElementById('flash-messages')) {
                        document.querySelector('.max-w-7xl.mx-auto.p-4').prepend(flashMessages);
                    }

                    // --- Refresh the Page ---
                    // Since the new data is only loaded server-side by the API call,
                    // we must force a page reload to trigger the client to get the new APP_STATE.
                    setTimeout(() => {
                        window.location.href = "{{ url_for('categories') }}";
                    }, 1500); // Redirect after a short delay so the user sees the success message

                } else {
                    // Error response from backend API
                    const errorMsg = `
                        <div class="p-3 rounded-lg text-sm bg-red-100 text-red-800" role="alert">
                            ❌ Processing error: ${result.message || 'Unknown error occurred in backend.'}
                        </div>
                    `;
                    flashMessages.innerHTML = errorMsg;
                    if (!document.getElementById('flash-messages')) {
                        document.querySelector('.max-w-7xl.mx-auto.p-4').prepend(flashMessages);
                    }
                }

            } catch (error) {
                // Network or fetch failure
                const networkErrorMsg = `
                    <div class="p-3 rounded-lg text-sm bg-red-100 text-red-800" role="alert">
                        ⚠️ Network error during upload: ${error.message}
                    </div>
                `;
                flashMessages.innerHTML = networkErrorMsg;
                if (!document.getElementById('flash-messages')) {
                    document.querySelector('.max-w-7xl.mx-auto.p-4').prepend(flashMessages);
                }
            } finally {
                // 5. Hide Loading State
                button.classList.remove('hidden');
                spinner.classList.add('hidden');
                form.querySelector('input[type="file"]').disabled = false;
            }
        });
    </script>
{% endblock content %}
""",
    "categories": """
{% extends "base" %}
{% block title %}Select Categories{% endblock title %}
{% block content %}
    <h2 class="text-3xl font-bold text-gray-900 mb-6 border-b pb-2">Step 2: Choose Target Categories</h2>
    <p class="text-gray-600 mb-8">Select one or more categories that best align with your product or campaign objectives. Categories are case-insensitive for matching.</p>

    {% set categories = CATEGORIES_LIST %}

    <form method="POST" action="{{ url_for('influencers') }}">
        <div class="grid md:grid-cols-3 gap-6">
            {% for id, name, description in categories %}
                <label for="{{ id }}" class="p-4 bg-white rounded-xl shadow-md border-2 border-gray-200 hover:border-indigo-500 cursor-pointer flex flex-col justify-between card">
                    <div class="flex items-start">
                        <input id="{{ id }}" type="checkbox" name="category" value="{{ name }}" class="h-5 w-5 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500 mt-1 mr-3">
                        <div>
                            <span class="text-lg font-semibold text-gray-900 capitalize">{{ name }}</span>
                            <p class="text-sm text-gray-500 mt-1">{{ description }}</p>
                        </div>
                    </div>
                </label>
            {% endfor %}
        </div>

        <div class="mt-10 text-center">
            <button type="submit" class="py-3 px-8 bg-indigo-600 text-white font-semibold rounded-full shadow-lg hover:bg-indigo-700 transition duration-200 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-indigo-500 focus:ring-opacity-50">
                Find Influencers
            </button>
        </div>
    </form>
{% endblock content %}
""",
    "influencers": """
{% extends "base" %}
{% block title %}Top Matched Influencers{% endblock title %}
{% block content %}
    <h2 class="text-3xl font-bold text-gray-900 mb-6 border-b pb-2">Step 3: Top Matched Influencers</h2>
    <p class="text-lg font-medium text-indigo-600 mb-6">
        {% if APP_STATE['data_loaded'] %}
            Results generated from loaded data.
        {% else %}
            ⚠️ Data not loaded. Please return to the Data step.
        {% endif %}
    </p>

    <div class="flex flex-wrap items-center gap-2 mb-6 p-3 bg-gray-50 rounded-lg">
        <span class="font-semibold text-gray-700">Filters Applied:</span>
        {% if selected_categories %}
            {% for cat in selected_categories %}
                <span class="text-sm font-medium px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full capitalize">{{ cat }}</span>
            {% endfor %}
        {% else %}
            <span class="text-sm font-medium px-3 py-1 bg-gray-200 text-gray-700 rounded-full">All Categories</span>
        {% endif %}
        <a href="{{ url_for('categories') }}" class="ml-auto text-sm text-indigo-500 hover:text-indigo-700 font-medium">
            Change Categories &rarr;
        </a>
    </div>

    {% if influencers %}
        <div class="overflow-x-auto shadow-md rounded-lg">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-indigo-100">
                    <tr>
                        <th class="px-4 py-3 text-left text-xs font-medium text-indigo-700 uppercase tracking-wider">Rank</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-indigo-700 uppercase tracking-wider">Influencer</th>
                        <th class="px-4 py-3 text-left text-xs font-medium text-indigo-700 uppercase tracking-wider">Category</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-indigo-700 uppercase tracking-wider">Followers</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-indigo-700 uppercase tracking-wider">Likes</th>
                        <th class="px-4 py-3 text-right text-xs font-medium text-indigo-700 uppercase tracking-wider">Comments</th>
                        <!-- UPDATED: Renamed header to just Sentiment -->
                        <th class="px-4 py-3 text-right text-xs font-medium text-indigo-700 uppercase tracking-wider">Sentiment</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% for influencer in influencers %}
                    <tr class="hover:bg-gray-50">
                        <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">{{ influencer.rank }}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-900 capitalize">{{ influencer.influencer }}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-indigo-600 capitalize">{{ influencer.category }}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700 text-right">{{ format_number(influencer.followers) }}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700 text-right">{{ format_number(influencer.likes) }}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700 text-right">{{ format_number(influencer.comments) }}</td>
                        <!-- UPDATED: Display only the emoji -->
                        <td class="px-4 py-3 whitespace-nowrap text-lg text-right">
                            {{ get_sentiment_emoji(influencer['sentiment_score'] | default(0)) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    {% else %}
        <div class="text-center p-10 bg-yellow-50 rounded-xl border border-yellow-200">
            <p class="text-lg text-yellow-800 font-medium">No influencers match the selected categories or no data is currently loaded.</p>
            <a href="{{ url_for('categories') }}" class="mt-4 inline-block text-indigo-600 hover:underline">Return to Category Selection</a>
        </div>
    {% endif %}

{% endblock content %}
""",
}

# Configure Jinja Environment with DictLoader
app.jinja_env = Environment(loader=DictLoader(TEMPLATES), autoescape=True)
# Register utility functions directly with the Jinja environment
app.jinja_env.globals.update(
    url_for=url_for,
    get_flashed_messages=get_flashed_messages,
    APP_STATE=APP_STATE,
    CATEGORIES_LIST=CATEGORIES_LIST,
    format_number=format_number,
    get_sentiment_emoji=get_sentiment_emoji,
    DEFAULT_CSV_PATH=DEFAULT_CSV_PATH,
)


# --- Flask Routes ---


@app.route("/")
def landing():
    """Renders the main landing page."""
    template = app.jinja_env.get_template("landing")
    return template.render()


# --- NEW API ROUTE (Simulated Backend) ---
@app.route("/api/process-txt", methods=["POST"])
def process_txt_api():
    """Simulates sending the TXT file to a backend, processing it,
    and returning a new CSV to replace the old data."""

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return (
            jsonify(
                {"success": False, "message": "Invalid file name or type. Must be .txt"}
            ),
            400,
        )

    # 1. Save file locally (optional, but good practice for logging/cleanup)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 2. Simulate Backend Processing (This is where your AI/Backend logic would run)
    print(f"DEBUG: Simulating AI processing of {filename}. Waiting 3 seconds...")
    time.sleep(3)  # Simulate a long processing job

    # 3. Simulate Backend Returning a NEW CSV Payload
    success, message = load_data_from_csv_content(MOCK_API_CSV_RESPONSE)

    # 4. Cleanup the uploaded file (optional)
    os.remove(filepath)

    if success:
        # The data is updated server-side in APP_STATE.
        # The frontend will refresh to display the new data.
        return (
            jsonify({"success": True, "message": "Data processed and replaced."}),
            200,
        )
    else:
        return (
            jsonify(
                {"success": False, "message": f"Data replacement failed: {message}"}
            ),
            500,
        )


@app.route("/upload", methods=["GET"])
def upload_data():
    """Renders the upload page and attempts automatic CSV load on GET request."""

    # --- GET REQUEST: Attempt Automatic CSV Load (Influencer Data) ---
    if not APP_STATE["data_loaded"]:
        success, message = load_data_from_path(DEFAULT_CSV_PATH)
        if success:
            flash(message, "success")
        else:
            flash(message, "error")

    template = app.jinja_env.get_template("upload")
    return template.render()


@app.route("/categories")
def categories():
    """Renders the 9 categories selection page."""
    template = app.jinja_env.get_template("categories")
    return template.render()


@app.route("/influencers", methods=["GET", "POST"])
def influencers():
    """Renders the influencer results page with filtering and local ranking."""

    if not APP_STATE["data_loaded"]:
        flash("Please load data first. Attempting automatic load now.", "error")
        # Try to load default data again if none is present
        success, _ = load_data_from_path(DEFAULT_CSV_PATH)
        if not success:
            return redirect(url_for("upload_data"))  # Redirect if load still fails

    all_influencers = APP_STATE["influencer_data"]
    selected_categories = []
    filtered_influencers = []

    if request.method == "POST":
        selected_categories = [c.lower() for c in request.form.getlist("category")]

        if selected_categories:
            filtered_influencers = [
                i
                for i in all_influencers
                if "category" in i and str(i["category"]).lower() in selected_categories
            ]
        else:
            filtered_influencers = all_influencers
    else:
        filtered_influencers = all_influencers

    # --- Assign Local Rank after Filtering ---
    final_ranked_influencers = []
    for index, influencer in enumerate(filtered_influencers):
        ranked_influencer = influencer.copy()
        ranked_influencer["rank"] = index + 1
        final_ranked_influencers.append(ranked_influencer)

    template = app.jinja_env.get_template("influencers")
    return template.render(
        influencers=final_ranked_influencers,
        selected_categories=selected_categories,
    )


if __name__ == "__main__":
    app.run(debug=True)
