import numpy as np
import pandas as pd
import gdown
from flask import Flask, jsonify, request
from statsmodels.tsa.arima.model import ARIMA

# ======================================================
# CONFIG
# ======================================================

USD_LINK = "https://drive.google.com/uc?id=1-M3n5CPuR-litZZw--nc7J2cAZlBvVD5"
EUR_LINK = "https://drive.google.com/uc?id=1Ik640rOV-8Sxbxh_BRcuT0R-QI00pRbs"
CNY_LINK = "https://drive.google.com/uc?id=1lT9c0uLCQZdFXCgyp-ZpJFLxju72ugXQ"

DATA_LINKS = {
    "USD": USD_LINK,
    "EUR": EUR_LINK,
    "CNY": CNY_LINK
}

ARIMA_CONFIG = {
    "USD": {
        "1d": {"order": (0,1,0), "horizon": 1},
        "5d": {"order": (0,1,0), "horizon": 5},
    },
    "EUR": {
        "1d": {"order": (0,1,0), "horizon": 1},
        "5d": {"order": (0,1,0), "horizon": 5},
    },
    "CNY": {
        "1d": {"order": (0,1,0), "horizon": 1},
        "5d": {"order": (0,1,0), "horizon": 5},
    }
}

TEST_SIZE = 365

# ======================================================
# UTILS
# ======================================================

def load_drive_csv(link):
    output = "temp.csv"
    gdown.download(link, output, quiet=True)
    df = pd.read_csv(output)

    # assume first column is date, second is close
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)
    df = df.rename(columns={df.columns[1]: "close"})
    df = df.sort_values(df.columns[0])
    df = df.set_index(df.columns[0])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()

    return df


# ======================================================
# ARIMA CORE 
# ======================================================

def arima_forecast_1d(y_train, y_test, order):
    history = y_train.values.tolist()
    y_test_list = y_test.values.tolist()

    preds = []

    for t in range(len(y_test_list)):
        model = ARIMA(np.array(history), order=order)
        fit = model.fit()

        fcast = fit.forecast()[0]
        preds.append(float(fcast))

        history.append(y_test_list[t])

    return pd.Series(preds, index=y_test.index)


def arima_forecast_5d(y_train, y_test, order, horizon):
    history = y_train.values.tolist()
    y_test_list = y_test.values.tolist()

    n_pred = len(y_test_list) - horizon
    preds = []

    for t in range(n_pred):
        model = ARIMA(np.array(history), order=order)
        fit = model.fit()

        fcast = fit.forecast(steps=horizon)
        preds.append(float(fcast[-1]))

        history.append(y_test_list[t])

    return pd.Series(preds, index=y_test.index[horizon:])


# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)

@app.route("/forecast", methods=["GET"])
def forecast():

    currency = request.args.get("currency", "USD").upper()
    horizon_key = request.args.get("horizon", "1d")

    if currency not in DATA_LINKS:
        return jsonify({"error": "Invalid currency"}), 400

    if horizon_key not in ["1d", "5d"]:
        return jsonify({"error": "Invalid horizon"}), 400

    # Load data
    df = load_drive_csv(DATA_LINKS[currency])

    y = df["close"]
    y_train = y.iloc[:-TEST_SIZE]
    y_test  = y.iloc[-TEST_SIZE:]

    cfg = ARIMA_CONFIG[currency][horizon_key]
    order = cfg["order"]
    horizon = cfg["horizon"]

    # Forecast
    if horizon_key == "1d":
        pred = arima_forecast_1d(y_train, y_test, order)
        actual = y_test
    else:
        pred = arima_forecast_5d(y_train, y_test, order, horizon)
        actual = y_test.iloc[horizon:]

    # JSON output (frontend-friendly)
    return jsonify({
        "currency": currency,
        "horizon": horizon_key,
        "order": order,
        "dates": pred.index.strftime("%Y-%m-%d").tolist(),
        "forecast": pred.round(4).tolist(),
        "actual": actual.loc[pred.index].round(4).tolist()
    })


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    app.run(debug=True)
