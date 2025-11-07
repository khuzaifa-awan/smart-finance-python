"""
to run this code use this command:


        uvicorn api:app --reload --host 0.0.0.0 --port 8000


"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import base64
from io import BytesIO
from datetime import date
import numpy as np
import logging


# ======================
# 1. FastAPI Init
# ======================
app = FastAPI(title="AI Financial Advisor API")

# Configure Gemini
genai.configure(api_key="AIzaSyA0BM3D7WmbJ4NiowMIob25vLi8hk4rpqo")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ======================
# 2. Load ML Model
# ======================
with open("savings_goal_clf.pkl", "rb") as f:
    savings_model = joblib.load(f)

FEATURES = [
    "income",
    "total_expenses",
    "goal_target_amount",
    "months_remaining",
    "monthly_saving_label_raw",
    "expense_ratio",
    "savings_rate",
    "financial_health_score",
    "total_potential_savings",
    "suggested_monthly_plan",
    "savings_is_synthetic",
    "age",
    "dependents",
    "occupation",
    "city_tier",
    "education",
    "desired_savings_percentage",
    "potential_savings_education",
]


# ======================
# 3. Request Schemas
# ======================
class SavingsRequest(BaseModel):
    income: float
    total_expenses: float
    goal_target_amount: float
    months_remaining: int
    monthly_saving_label_raw: float
    expense_ratio: float
    savings_rate: float
    financial_health_score: float
    total_potential_savings: float
    suggested_monthly_plan: float
    savings_is_synthetic: int
    age: int
    dependents: int
    occupation: str
    city_tier: str
    education: int
    desired_savings_percentage: float
    potential_savings_education: float


class StockRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date


# ======================
# 4. Helpers
# ======================
def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ======================
# 5. Endpoints
# ======================


# helper to make values JSON-safe
def to_native(x):
    """
    Convert numpy / pandas scalar types to native Python types.
    If x is an array-like, convert elements recursively.
    """
    try:
        # numpy scalar (has .item())
        if hasattr(x, "item"):
            return x.item()
        # numpy arrays / pandas Series -> convert to list of native
        if isinstance(x, (np.ndarray, list, tuple)):
            return [to_native(v) for v in x]
        return x
    except Exception:
        return x


@app.post("/predict_savings")
def predict_savings(request: SavingsRequest):
    try:
        # Build DataFrame with expected feature order
        input_df = pd.DataFrame([request.dict()], columns=FEATURES)

        # Raw model outputs (may be numpy types)
        pred_raw = savings_model.predict(input_df)[0]
        proba_raw = savings_model.predict_proba(input_df)[0][1]

        # Convert to native Python types (int/float)
        prediction = int(to_native(pred_raw))
        confidence = float(to_native(proba_raw))

        # Build prompt and call Gemini (use your call_gemini wrapper if you have one)
        prompt = f"""
        You are a financial advisor AI. Interpret the following profile and model prediction.

        User profile:
        {request.dict()}

        ML Prediction:
        - Achievable by cutting: {bool(prediction)}
        - Confidence: {confidence:.2f}

        Please provide:
        1. A short plain-language explanation.
        2. Practical savings or expense adjustment advice tailored to the profile.
        """
        # call_gemini should return a string. If you don't have it, use gemini_model.generate_content(...)
        try:
            resp = (
                gemini_model.generate_content(prompt)
                if gemini_model is not None
                else None
            )
            gemini_text = (
                getattr(resp, "text", str(resp))
                if resp is not None
                else "Gemini disabled or not configured."
            )
        except Exception as e:
            logging.exception("Gemini call failed")
            gemini_text = f"Gemini call failed: {e}"

        # Return only native python types
        return {
            "prediction": prediction,
            "confidence": confidence,
            "gemini_insights": gemini_text,
        }

    except Exception as e:
        logging.exception("predict_savings failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Stock Insights ----
@app.post("/stock_insights")
def stock_insights(request: StockRequest):
    try:
        stock = yf.download(
            request.ticker, start=request.start_date, end=request.end_date
        )
        if stock.empty:
            raise HTTPException(status_code=404, detail="No stock data found.")

        stock.reset_index(inplace=True)
        close = stock["Close"].squeeze()
        volume = stock["Volume"].squeeze()

        # Charts
        charts = []

        # 1. Closing Price
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=stock["Date"], y=close, label="Closing Price")
        plt.title(f"{request.ticker} Closing Price Over Time")
        charts.append(fig_to_base64())
        plt.close()

        # 2. Volume
        plt.figure(figsize=(10, 5))
        sns.barplot(x=stock["Date"], y=volume, color="orange")
        plt.title(f"{request.ticker} Trading Volume Over Time")
        plt.xticks(rotation=45)
        charts.append(fig_to_base64())
        plt.close()

        # 3. Moving Averages
        stock["MA20"] = close.rolling(20).mean()
        stock["MA50"] = close.rolling(50).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(stock["Date"], close, label="Close", color="blue")
        plt.plot(stock["Date"], stock["MA20"], label="MA20", color="red")
        plt.plot(stock["Date"], stock["MA50"], label="MA50", color="green")
        plt.legend()
        plt.title(f"{request.ticker} Moving Averages")
        charts.append(fig_to_base64())
        plt.close()

        # Key stats
        start_price = stock["Close"].iloc[0].item()
        end_price = stock["Close"].iloc[-1].item()
        avg_price = stock["Close"].mean().item()
        volatility = (stock["Close"].std() / stock["Close"].mean()).item()
        max_price = stock["Close"].max().item()
        min_price = stock["Close"].min().item()

        # Gemini
        prompt = f"""
        You are a financial assistant. Analyze {request.ticker} stock.
        Key stats:
        - Start price: {start_price:.2f}
        - End price: {end_price:.2f}
        - Average price: {avg_price:.2f}
        - Volatility: {volatility:.2%}
        - Max price: {max_price:.2f}
        - Min price: {min_price:.2f}

        Please provide:
        1. A plain-language summary.
        2. Risks and opportunities.
        3. Suggested next steps for an investor.
        """
        response = gemini_model.generate_content(prompt)

        return {
            "ticker": request.ticker,
            "charts": charts,  # Base64 PNGs
            "gemini_insights": response.text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
