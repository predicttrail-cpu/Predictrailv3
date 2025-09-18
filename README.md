# PredictTrail 🏃💨

PredictTrail is a web application designed to provide trail runners with a sophisticated, data-driven race plan for their upcoming events. By leveraging your personal Strava data and the specifics of your race course, PredictTrail generates a detailed strategy to help you perform at your best.

## 🌟 Key Features

*   **Strava Integration**: Securely connect your Strava account to allow PredictTrail to analyze your past running activities.
*   **In-Depth Athlete Profiling**: The application automatically generates a comprehensive profile of you as a trail runner. It identifies your strengths and weaknesses across various domains, such as:
    *   Climbing Power (VAM)
    *   Downhill Technique
    *   Aerobic Endurance on flats
    *   Fatigue Resistance
    *   Pacing Management
    *   This profile is visualized in an easy-to-understand radar chart.
*   **GPX Route Analysis**: Upload a GPX file of your race course. PredictTrail will parse the route, identify key segments (climbs, descents, flats), and display it on an interactive map.
*   **Personalized Race & Nutrition Plan**: Based on your unique profile and the course demands, the application generates a complete race plan, including:
    *   Predicted time and pace for each segment of the race.
    *   Estimated calorie expenditure.
    *   Personalized carbohydrate and hydration recommendations (in grams/hour and mL/hour).
*   **Weather Adjustments**: The plan can be adjusted based on the expected temperature for the race day, giving you a more accurate prediction.
*   **Interactive UI**: A modern, user-friendly interface allows you to easily connect your accounts, upload your files, and view your detailed plan.

## 🚀 How to Use

1.  Visit the web application.
2.  Connect your Strava account.
3.  Upload the GPX file for your race.
4.  Enter basic parameters like your weight and planned carbohydrate intake.
5.  Click "Calculate my plan" to receive your personalized race strategy!

## Local Development Setup

To run the PredictTrail application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/votre-utilisateur/PredictTrail.git
    cd PredictTrail
    ```

2.  **Set up the backend:**
    - Navigate to the `backend` directory:
      ```bash
      cd backend
      ```
    - Create and activate a Python virtual environment. This isolates the project dependencies.
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
      ```
    - Install the required dependencies:
      ```bash
      pip install -r requirements.txt
      ```

3.  **Configure Environment Variables:**
    - The application requires API keys for Strava. You will need to create a `.env` file inside the `backend` directory.
    - Create a file named `.env` in the `backend` directory and add the following content, replacing the placeholders with your actual Strava API credentials:
      ```
      STRAVA_CLIENT_ID=your_strava_client_id
      STRAVA_CLIENT_SECRET=your_strava_client_secret
      ```

4.  **Run the application:**
    - From the root directory of the project, execute the `run.sh` script:
      ```bash
      ./run.sh
      ```
    - The application will be available at `http://127.0.0.1:8000`.

## 🛠️ Technology Stack

*   **Backend**: Python with FastAPI
*   **Frontend**: HTML, Tailwind CSS, and vanilla JavaScript
*   **Mapping**: Leaflet.js
*   **Charting**: Chart.js
