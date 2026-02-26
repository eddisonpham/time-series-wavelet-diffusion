# data_downloader.py
# Synthetic Merton Jump Diffusion data generator

import os
import argparse
import numpy as np
import pandas as pd


def generate_merton_jump_diffusion(
    S0=1.10,
    mu=0.05,
    sigma=0.2,
    lam=0.1,
    m=-0.02,
    v=0.1,
    dt=1/1440,
    steps=1440
):
    prices = np.zeros(steps)
    prices[0] = S0

    kappa = np.exp(m + 0.5 * v**2) - 1

    for t in range(1, steps):
        z = np.random.normal()
        dW = np.sqrt(dt) * z

        poisson_jump = np.random.poisson(lam * dt)
        jump = 0.0
        if poisson_jump > 0:
            jump = np.sum(np.random.normal(m, v, poisson_jump))

        drift = (mu - lam * kappa - 0.5 * sigma**2) * dt
        diffusion = sigma * dW

        prices[t] = prices[t-1] * np.exp(drift + diffusion + jump)

    return prices


def create_ohlc(prices):
    df = pd.DataFrame({"Close": prices})
    df["Open"] = df["Close"].shift(1)
    df.loc[0, "Open"] = df.loc[0, "Close"]

    df["High"] = df[["Open", "Close"]].max(axis=1) + np.abs(np.random.normal(0, 0.0005, len(df)))
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.abs(np.random.normal(0, 0.0005, len(df)))

    df["Volume"] = np.random.randint(100, 1000, len(df))

    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair", type=str, default="EURUSD")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--month", type=int, default=1)

    parser.add_argument("--days", type=int, default=30)

    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--jump_mean", type=float, default=-0.02)
    parser.add_argument("--jump_std", type=float, default=0.1)

    args = parser.parse_args()

    minutes = 1440 * args.days
    dt = 1 / 1440

    prices = generate_merton_jump_diffusion(
        S0=1.10,
        mu=args.mu,
        sigma=args.sigma,
        lam=args.lam,
        m=args.jump_mean,
        v=args.jump_std,
        dt=dt,
        steps=minutes
    )

    df = create_ohlc(prices)

    start_date = pd.Timestamp(f"{args.year}-{args.month:02d}-01 00:00:00")

    timestamps = pd.date_range(start_date, periods=minutes, freq="min")

    df.insert(0, "Date", timestamps.date.astype(str))
    df.insert(1, "Time", timestamps.time.astype(str))

    os.makedirs("data", exist_ok=True)

    output_path = f"data/{args.pair.upper()}_{args.year}_{args.month:02d}_M1.csv"
    df.to_csv(output_path, index=False)

    print(f"Synthetic Merton Jump Diffusion data saved to: {output_path}")
    print(f"Rows generated: {len(df)}")


if __name__ == "__main__":
    main()