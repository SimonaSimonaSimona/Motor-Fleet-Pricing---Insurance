
# ============================================================
# INSURANCE CLAIMS ANALYSIS AND PRICING – 2026
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. ASSUMPTIONS --------------------------

FILE_NAME = "Case Protector Analyst 2026 .xlsx"
OUTPUT_FOLDER = "Claims_Analysis_Results"

INFLATION = 0.05
FORECAST_YEAR = 2026
FORECAST_EXPOSURE = 100

# PROPOSED_PREMIUM = 365_000
# COMPETITOR_PREMIUM = 450_000
# TARGET_LOSS_RATIO = 0.72
COMPETITOR_PREMIUM = 450_000
ADMIN_COST = 0

N_SIMULATIONS = 100_000
RANDOM_SEED = 42

# Historical exposure in vehicle-years
EXPOSURE = {
    2022: 1,
    2023: 2,
    2024: 87,
    2025: 100
}

# Historical annual costs supplied in the case
EXPECTED_HISTORICAL_COSTS = {
    2022: 11_586,
    2023: 16_417,
    2024: 217_901,
    2025: 211_863
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

np.random.seed(RANDOM_SEED)


#  2. LOAD DATA -------------------------

def normalise_column(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


excel = pd.ExcelFile(FILE_NAME)

claims = None
claims_sheet = None

for sheet in excel.sheet_names:
    temp = pd.read_excel(FILE_NAME, sheet_name=sheet)
    column_map = {
        normalise_column(c): c for c in temp.columns
    }

    if (
        "year" in column_map
        and "claimcost" in column_map
    ):
        claims = temp.copy()
        claims_sheet = sheet
        break

if claims is None:
    raise ValueError(
        "Could not find a sheet containing Year and Claim cost columns."
    )

column_map = {
    normalise_column(c): c for c in claims.columns
}

year_col = column_map["year"]
cost_col = column_map["claimcost"]
date_col = column_map.get("claimdate")

claims = claims.rename(columns={
    year_col: "year",
    cost_col: "claim_cost"
})

if date_col:
    claims = claims.rename(columns={date_col: "claim_date"})
    claims["claim_date"] = pd.to_datetime(
        claims["claim_date"], errors="coerce"
    )
else:
    claims["claim_date"] = pd.NaT

claims["year"] = pd.to_numeric(
    claims["year"], errors="coerce"
)

claims["claim_cost"] = pd.to_numeric(
    claims["claim_cost"], errors="coerce"
)

# Audit missing and invalid values before removing anything
missing_years = claims["year"].isna().sum()
missing_costs = claims["claim_cost"].isna().sum()
negative_costs = (claims["claim_cost"] < 0).sum()

claims["year"] = claims["year"].astype("Int64")

if missing_years > 0 or missing_costs > 0:
    warnings.warn(
        f"Missing years: {missing_years}; "
        f"missing costs: {missing_costs}. "
        "Rows with missing values will be excluded."
    )

if negative_costs > 0:
    raise ValueError("Negative claim costs found. Please investigate.")

claims = claims.dropna(subset=["year", "claim_cost"]).copy()
claims["year"] = claims["year"].astype(int)

# Keep the four historical years
claims = claims[claims["year"].isin(EXPOSURE)].copy()

# Check whether claim dates agree with recorded years
date_year_mismatches = (
    claims["claim_date"].notna()
    & (claims["claim_date"].dt.year != claims["year"])
).sum()

# Zero-cost claims are retained as claims.
# This assumes each row represents a genuine reported claim.


#  3. DATA AUDIT ------------------------

annual = (
    claims.groupby("year")
    .agg(
        claim_count=("claim_cost", "size"),
        total_claim_cost=("claim_cost", "sum"),
        average_severity=("claim_cost", "mean"),
        median_severity=("claim_cost", "median"),
        maximum_claim=("claim_cost", "max")
    )
)

annual["exposure"] = pd.Series(EXPOSURE)
annual["frequency"] = (
    annual["claim_count"] / annual["exposure"]
)

annual["cost_difference_vs_case"] = (
    annual["total_claim_cost"]
    - pd.Series(EXPECTED_HISTORICAL_COSTS)
)

audit = pd.DataFrame({
    "Check": [
        "Source Excel sheet",
        "Number of claim records",
        "Total historical exposure",
        "Total historical claim cost",
        "Missing years",
        "Missing claim costs",
        "Negative claim costs",
        "Claim date/year mismatches",
        "Total claims including zero-cost claims"
    ],
    "Result": [
        claims_sheet,
        len(claims),
        sum(EXPOSURE.values()),
        claims["claim_cost"].sum(),
        missing_years,
        missing_costs,
        negative_costs,
        date_year_mismatches,
        len(claims)
    ]
})

print("\nDATA AUDIT")
print(audit.to_string(index=False))

print("\nANNUAL RECONCILIATION")
print(annual.round(2).to_string())

if not np.allclose(
    annual["total_claim_cost"].reindex(EXPOSURE).fillna(0).values,
    pd.Series(EXPECTED_HISTORICAL_COSTS).reindex(EXPOSURE).values,
    atol=1
):
    warnings.warn(
        "Individual claim costs do not reconcile with the "
        "annual costs supplied in the case. Check the data."
    )


#  4. INFLATION -------------------------

annual["inflation_factor"] = (
    1 + INFLATION
) ** (FORECAST_YEAR - annual.index)

annual["trended_claim_cost"] = (
    annual["total_claim_cost"]
    * annual["inflation_factor"]
)

annual["trended_severity"] = (
    annual["average_severity"]
    * annual["inflation_factor"]
)

annual["trended_cost_per_vehicle"] = (
    annual["trended_claim_cost"]
    / annual["exposure"]
)

claims["inflation_factor"] = (
    1 + INFLATION
) ** (FORECAST_YEAR - claims["year"])

claims["trended_claim_cost"] = (
    claims["claim_cost"]
    * claims["inflation_factor"]
)


# 5. FREQUENCY AND SEVERITY -----------

total_claims = len(claims)
total_exposure = sum(EXPOSURE.values())

# Exposure-weighted historical frequency
frequency = total_claims / total_exposure

# Inflation-adjusted average severity
average_severity = claims["trended_claim_cost"].mean()

# Expected claim count for 2026
expected_claims = frequency * FORECAST_EXPOSURE

# Expected aggregate claims cost
pure_premium = (
    frequency * average_severity * FORECAST_EXPOSURE
)

# Equivalent calculation using total trended claims cost
alternative_pure_premium = (
    claims["trended_claim_cost"].sum()
    / total_exposure
    * FORECAST_EXPOSURE
)

assert np.isclose(pure_premium, alternative_pure_premium)

# Zero administrative costs; no assumed loss ratio
technical_premium = pure_premium + ADMIN_COST
PROPOSED_PREMIUM = technical_premium

print("\nPRICING RESULTS")
print(f"Total claims: {total_claims}")
print(f"Total exposure: {total_exposure}")
print(f"Historical frequency: {frequency:.4f}")
print(f"Expected 2026 claims: {expected_claims:.2f}")
print(f"Inflation-adjusted severity: {average_severity:,.2f} SEK")
print(f"Expected 2026 claims cost: {pure_premium:,.2f} SEK")
print(f"Technical premium: {technical_premium:,.2f} SEK")
print(f"Proposed premium: {PROPOSED_PREMIUM:,.2f} SEK")
print(f"Competitor premium: {COMPETITOR_PREMIUM:,.2f} SEK")


# ---------------------- 6. PREMIUM COMPARISON ----------------

pricing_comparison = pd.DataFrame({
    "Measure": [
        "Expected claims cost",
        "Premium at target loss ratio",
        "Proposed premium",
        "Competitor premium"
    ],
    "Amount_SEK": [
        pure_premium,
        technical_premium,
        PROPOSED_PREMIUM,
        COMPETITOR_PREMIUM
    ]
})

implied_loss_ratio = pure_premium / PROPOSED_PREMIUM
premium_discount = (
    1 - PROPOSED_PREMIUM / COMPETITOR_PREMIUM
)

print(f"\nImplied loss ratio at proposed premium: {implied_loss_ratio:.1%}")
print(f"Discount to competitor: {premium_discount:.1%}")


# 7. MONTE CARLO SIMULATION ------------

# Compound Poisson model:
# Claim count ~ Poisson(expected frequency x exposure)
# Severity is sampled from the historical inflation-adjusted claims.

trended_severities = claims["trended_claim_cost"].to_numpy()

if len(trended_severities) == 0:
    raise ValueError("No valid claim severities available.")

simulated_counts = np.random.poisson(
    lam=expected_claims,
    size=N_SIMULATIONS
)

max_count = int(simulated_counts.max())

# Generate severity samples in a matrix and sum the relevant claims
sampled_severities = np.random.choice(
    trended_severities,
    size=(N_SIMULATIONS, max_count),
    replace=True
)

column_positions = np.arange(max_count)

simulated_costs = (
    sampled_severities
    * (column_positions < simulated_counts[:, None])
).sum(axis=1)

mc_percentiles = np.percentile(
    simulated_costs, [50, 75, 90, 95, 99]
)

mc_results = pd.DataFrame({
    "Measure": [
        "Mean simulated claims cost",
        "Median",
        "75th percentile",
        "90th percentile",
        "95th percentile",
        "99th percentile",
        "Probability claims cost below proposed premium",
        "Probability claims cost below competitor premium"
    ],
    "Result": [
        simulated_costs.mean(),
        mc_percentiles[0],
        mc_percentiles[1],
        mc_percentiles[2],
        mc_percentiles[3],
        mc_percentiles[4],
        np.mean(simulated_costs < PROPOSED_PREMIUM),
        np.mean(simulated_costs < COMPETITOR_PREMIUM)
    ]
})

print("\nMONTE CARLO RESULTS")
print(mc_results.to_string(index=False))


#  8. PLOT 1: EXPOSURE AND CLAIM COUNTS --

years = list(EXPOSURE.keys())

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(
    years,
    [EXPOSURE[y] for y in years],
    alpha=0.65,
    label="Vehicle exposure"
)
ax1.set_xlabel("Accident year")
ax1.set_ylabel("Vehicle-years")
ax1.set_xticks(years)

ax2 = ax1.twinx()
ax2.plot(
    years,
    [annual.loc[y, "claim_count"] for y in years],
    marker="o",
    linewidth=2,
    label="Number of claims"
)
ax2.set_ylabel("Number of claims")

ax1.set_title("Historical Exposure and Number of Claims")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/01_exposure_and_claim_counts.png",
    dpi=300
)
plt.close(fig)


#  9. PLOT 2: CLAIM FREQUENCY ------------

fig, ax = plt.subplots(figsize=(9, 5))

frequencies = [
    annual.loc[y, "frequency"] * 100 for y in years
]

ax.bar(years, frequencies)
ax.axhline(
    frequency * 100,
    linestyle="--",
    label=f"Pooled frequency: {frequency:.1%}"
)

ax.set_title("Historical Claims Frequency")
ax.set_xlabel("Accident year")
ax.set_ylabel("Claims per 100 vehicle-years")
ax.set_xticks(years)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/02_claim_frequency.png",
    dpi=300
)
plt.close(fig)


#  10. PLOT 3: CLAIM COSTS ----------------

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(years))
width = 0.36

actual_costs = [
    annual.loc[y, "total_claim_cost"] for y in years
]

trended_costs = [
    annual.loc[y, "trended_claim_cost"] for y in years
]

ax.bar(
    x - width / 2,
    actual_costs,
    width,
    label="Actual claims cost"
)

ax.bar(
    x + width / 2,
    trended_costs,
    width,
    label="Trended to 2026"
)

ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel("Claims cost (SEK)")
ax.set_title("Historical Claims Costs: Actual vs Inflation-adjusted")
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/03_claim_costs.png",
    dpi=300
)
plt.close(fig)


#  11. PLOT 4: CLAIM SEVERITY --------------

fig, ax = plt.subplots(figsize=(9, 5))

actual_severity = [
    annual.loc[y, "average_severity"] for y in years
]

trended_severity = [
    annual.loc[y, "trended_severity"] for y in years
]

ax.plot(
    years,
    actual_severity,
    marker="o",
    label="Actual average severity"
)

ax.plot(
    years,
    trended_severity,
    marker="o",
    linestyle="--",
    label="Average severity trended to 2026"
)

ax.set_title("Average Claim Severity")
ax.set_xlabel("Accident year")
ax.set_ylabel("Average claim cost (SEK)")
ax.set_xticks(years)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/04_claim_severity.png",
    dpi=300
)
plt.close(fig)


#  12. PLOT 5: COST PER VEHICLE ------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    years,
    [
        annual.loc[y, "trended_cost_per_vehicle"]
        for y in years
    ],
    label="Trended claims cost per vehicle"
)

ax.axhline(
    pure_premium / FORECAST_EXPOSURE,
    linestyle="--",
    label="2026 expected cost per vehicle"
)

ax.set_title("Inflation-adjusted Claims Cost per Vehicle")
ax.set_xlabel("Accident year")
ax.set_ylabel("Claims cost per vehicle (SEK)")
ax.set_xticks(years)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/05_cost_per_vehicle.png",
    dpi=300
)
plt.close(fig)


#  13. PLOT 6: CLAIM SEVERITY DISTRIBUTION --

fig, ax = plt.subplots(figsize=(9, 5))

positive_severities = claims.loc[
    claims["trended_claim_cost"] > 0,
    "trended_claim_cost"
]

ax.hist(
    positive_severities,
    bins=25,
    edgecolor="black"
)

ax.set_title("Distribution of Positive Inflation-adjusted Claim Severities")
ax.set_xlabel("Claim severity (SEK)")
ax.set_ylabel("Number of claims")

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/06_severity_distribution.png",
    dpi=300
)
plt.close(fig)


#  14. PLOT 7: MONTE CARLO ------------------

fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    simulated_costs,
    bins=60,
    density=True,
    alpha=0.7,
    edgecolor="black"
)

ax.axvline(
    pure_premium,
    linestyle="--",
    linewidth=2,
    label=f"Expected claims cost: {pure_premium:,.0f} SEK"
)

ax.axvline(
    PROPOSED_PREMIUM,
    linestyle="-.",
    linewidth=2,
    label=f"Proposed premium: {PROPOSED_PREMIUM:,.0f} SEK"
)

ax.axvline(
    COMPETITOR_PREMIUM,
    linestyle=":",
    linewidth=2,
    label=f"Competitor premium: {COMPETITOR_PREMIUM:,.0f} SEK"
)

ax.set_title("Monte Carlo Distribution of 2026 Aggregate Claims Cost")
ax.set_xlabel("Annual aggregate claims cost (SEK)")
ax.set_ylabel("Probability density")
ax.legend()

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/07_monte_carlo_distribution.png",
    dpi=300
)
plt.close(fig)


#  15. PLOT 8: PREMIUM COMPARISON -----------

fig, ax = plt.subplots(figsize=(9, 5))

labels = [
    "Expected claims cost",
    #"Premium at 72% loss ratio",
    "Proposed premium",
    "Competitor premium"
]

values = [
    pure_premium,
    #technical_premium,
    PROPOSED_PREMIUM,
    COMPETITOR_PREMIUM
]

bars = ax.bar(labels, values)

ax.set_title("2026 Premium Comparison")
ax.set_ylabel("SEK per 100 vehicles")
ax.tick_params(axis="x", rotation=15)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,.0f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(
    f"{OUTPUT_FOLDER}/08_premium_comparison.png",
    dpi=300
)
plt.close(fig)


#  16. EXPORT RESULTS TO EXCEL -------------

summary = pd.DataFrame({
    "Metric": [
        "Historical claims",
        "Historical exposure",
        "Historical frequency",
        "Expected claims in 2026",
        "Inflation assumption",
        "Average trended severity",
        "Expected 2026 claims cost",
        "Technical premium",
        "Proposed premium",
        "Competitor premium",
        "Implied loss ratio",
        "Discount to competitor"
    ],
    "Value": [
        total_claims,
        total_exposure,
        frequency,
        expected_claims,
        INFLATION,
        average_severity,
        pure_premium,
        technical_premium,
        PROPOSED_PREMIUM,
        COMPETITOR_PREMIUM,
        implied_loss_ratio,
        premium_discount
    ]
})

output_excel = f"{OUTPUT_FOLDER}/Claims_Analysis_Results.xlsx"

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    annual.reset_index().to_excel(
        writer, sheet_name="Annual Analysis", index=False
    )
    audit.to_excel(writer, sheet_name="Data Audit", index=False)
    pricing_comparison.to_excel(
        writer, sheet_name="Premium Comparison", index=False
    )
    mc_results.to_excel(
        writer, sheet_name="Monte Carlo", index=False
    )
    claims_export = claims.copy()
    claims_export["claim_date"] = claims_export["claim_date"].dt.strftime("%Y-%m-%d")
    claims_export.to_excel(writer, sheet_name="Claim Data", index=False
    )

# Save simulated costs separately in the workbook
with pd.ExcelWriter(
    output_excel,
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace"
) as writer:
    pd.DataFrame({
        "Simulated aggregate claims cost": simulated_costs
    }).to_excel(
        writer,
        sheet_name="Simulation Data",
        index=False
    )


print("\nINDEPENDENT VALIDATION")

# Check historical claims
assert total_claims == 68
assert total_exposure == 190

# Check frequency
assert np.isclose(frequency, 68 / 190)

# Check expected claims
assert np.isclose(
    expected_claims,
    frequency * 100
)

# Check expected claims cost
assert np.isclose(
    pure_premium,
    claims["trended_claim_cost"].sum()
    / total_exposure * 100
)

# Check technical premium
assert np.isclose(
    technical_premium,
    pure_premium + ADMIN_COST
)

# Check simulation mean against expected cost
assert abs(
    simulated_costs.mean() - pure_premium
) / pure_premium < 0.02

print("All independent validation checks passed.")


# 17. END ----------------------------

print("\n" + "=" * 55)
print("ANALYSIS COMPLETED")
print("=" * 55)

print(f"\nResults saved in: {OUTPUT_FOLDER}")
print(f"Excel workbook: {output_excel}")

print("\nPlots created:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    if filename.endswith(".png"):
        print(" -", filename)

print("\nRemember:")
print("- Administrative costs are assumed to be zero.")
print("- The Monte Carlo model assumes a Poisson claim count.")
print("- Historical severities are used as the simulation distribution.")
print("- Results assume the historical portfolio is representative of 2026.")
