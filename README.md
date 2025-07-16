Acne Detection Web App

This is a skin analysis web application that uses deep learning to detect acne severity (Clear Skin, Mild Acne, Severe Acne) from facial images. The project includes a React frontend and a FastAPI backend connected to a trained acne detection model.

Features
1. Upload or capture face images  
2. Classifies acne into Clear, Mild, or Severe  
3. Displays customized skin-care recommendations  
4. Real-time prediction with FastAPI backend  
5. Simple and intuitive React-based UI  

Tech Stack
| Part               | Technology            |
|--------------------|------------------------
| Frontend           | ReactJS               |
| Backend            | FastAPI (Python)      |
| ML Model           | Keras / TensorFlow    |
| Image Processing   | OpenCV, NumPy         |
| Package Management | pip + npm             |

How to Run the Project
1. Clone the Repository

bash
git clone https://github.com/Bhagyashree038/acne-detection.git
cd acne-detection

2. Run the Backend (FastAPI + Python)

bash
cd backend
python -m venv venv
venv\Scripts\activate        # For Windows users
pip install -r requirements.txt
uvicorn app:app --reload

3. Run the Frontend (React)

bash
cd ../frontend/acne-detection
npm install
npm start


> The React frontend runs at: http://localhost:3000

4. Using the App

- Open browser at http://localhost:3000
- Upload or capture an image
- The backend predicts the acne severity
- The result + recommendations are displayed on screen

---

Troubleshooting

- Model file missing?
  Make sure your trained model file is in `backend/model/` and the path is correctly loaded in `app.py`.

- CORS error?
  Add this to your FastAPI app:

  ```python
  from fastapi.middleware.cors import CORSMiddleware

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

  Version mismatch?
  - Python: 3.8+  
  - Node: v16+ recommended

---

Model Info

- Trained on ~2000 labeled acne images
- Classification Labels:
  - `0`: Clear Skin
  - `1`: Mild Acne
  - `2`: Severe Acne
- Metrics used: Accuracy, F1 Score
