# Setup Instructions

1. Create a `.env` file in the project folder and paste this exact content inside:
```env
GOOGLE_API_KEY=
GEMINI_CHAT_MODEL=gemini-2.5-flash
LOCAL_HF_MODEL_PATH=
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=admin123

```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

4. Install requirements:
```
pip install -r requirement.txt
```
5. Prepare the knowledge base:
```
python prepare_kb.py
```

6. Run the application:
```
streamlit run app.py
```
