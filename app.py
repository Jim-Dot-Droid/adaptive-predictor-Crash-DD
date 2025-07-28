import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df['multiplier'].tolist()

def normalize_input(value):
    """Convert percentage input to multiplier if needed."""
    if value > 10:  # assume it's percentage
        return value / 100
    return value

def compute_improved_confidence(data, threshold=2.0, trend_window=10):
    if not data:
        return 0.5, 0.5
    
    data = np.array(data)
    n = len(data)

    # Weighted base frequency
    weights = np.linspace(0.5, 1.0, n)
    base_score = np.average((data > threshold).astype(int), weights=weights)

    # Recent trend
    recent = data[-trend_window:] if n >= trend_window else data
    trend_score = np.mean(recent > threshold) if len(recent) > 0 else 0.5

    # Streak detection
    streak = 1
    for i in range(n-2, -1, -1):
        if (data[i] > threshold and data[i+1] > threshold) or (data[i] <= threshold and data[i+1] <= threshold):
            streak += 1
        else:
            break
    streak_impact = min(streak * 0.01, 0.1)
    streak_score = streak_impact if data[-1] <= threshold else -streak_impact

    # Combine
    combined = (0.6 * base_score) + (0.3 * trend_score) + (0.1 * (0.5 + streak_score))

    # Volatility adjustment
    volatility = np.std(data)
    if volatility > 2:
        combined *= 0.9

    # Normalize for small data (cap confidence)
    if n < 20:
        combined = 0.5 + (combined - 0.5) * 0.5  # compress range
        combined = max(0.35, min(combined, 0.65))  # cap between 35%-65%

    combined = max(0, min(combined, 1))
    return combined, 1 - combined

def main():
    st.title("Crash Game Predictor (Improved)")
    st.write("Upload a CSV or enter values manually. Supports multipliers and percentages (e.g., 187 for 1.87x).")

    uploaded_file = st.file_uploader("Upload multipliers CSV", type=["csv"])
    data = []
    if uploaded_file:
        data = load_csv(uploaded_file)
        st.success(f"Loaded {len(data)} multipliers.")

    st.subheader("Manual Input")
    new_val = st.text_input("Enter a new multiplier or percentage (e.g., 1.87 or 187)")
    if st.button("Add"):
        try:
            val = float(new_val)
            val = normalize_input(val)
            data.append(val)
            st.success(f"Added {val}x")
        except:
            st.error("Invalid number.")

    if data:
        above_conf, under_conf = compute_improved_confidence(data)
        st.subheader("Prediction")
        if above_conf > under_conf:
            st.write(f"Prediction: **Above 200%** ({above_conf:.1%} confidence)")
        else:
            st.write(f"Prediction: **Under 200%** ({under_conf:.1%} confidence)")
    else:
        st.write("Add data to get prediction.")

if __name__ == "__main__":
    main()
