"""Central configuration: paths, splits, and the rupee cost model.

Every number a reviewer might challenge lives here, with its justification,
rather than being scattered as magic constants through the code.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"          # put IEEE-CIS CSVs here
ARTIFACTS = ROOT / "artifacts"    # models, plots, metric tables
ARTIFACTS.mkdir(exist_ok=True)

TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
IDENTITY_CSV = DATA_DIR / "train_identity.csv"

# --- temporal split --------------------------------------------------------
# Fractions of the TransactionDT-sorted data. Never random: a random split
# trains on transactions occurring after those it is tested on, which leaks
# the future and inflates every score.
TRAIN_FRAC = 0.70
CALIB_FRAC = 0.10   # sits between train and test; used only to fit calibration
TEST_FRAC = 0.20

SEED = 42

# --- rupee cost model ------------------------------------------------------
# The dataset is US (Vesta) card data. We apply an Indian merchant cost model
# to it. This is an explicit, disclosed assumption -- see ARCH.md.
#
# Amounts in IEEE-CIS (TransactionAmt) are USD. We convert at a fixed rate so
# every figure we report is in rupees and comparable across policies.
USD_TO_INR = 88.0

# Blocking a GENUINE transaction (false positive) costs:
#   the margin on the lost sale, plus the customer's remaining lifetime value.
MERCHANT_MARGIN_RATE = 0.25   # 25% gross margin -- typical Indian D2C
CUSTOMER_LTV_INR = 2500.0     # remaining expected value of a lost customer

# Allowing a FRAUD transaction (false negative) costs:
#   the transaction amount (goods gone, money clawed back) plus a chargeback fee.
CHARGEBACK_FEE_INR = 1200.0

# Sensitivity analysis varies each of the three parameters above by +/- this.
SENSITIVITY_RANGE = 0.30

# Transactions above this amount are never auto-blocked; they go to a human.
AUTO_BLOCK_LIMIT_INR = 50_000.0


def cost_of_blocking_genuine(amount_inr: float) -> float:
    """False positive: we declined a real customer."""
    return amount_inr * MERCHANT_MARGIN_RATE + CUSTOMER_LTV_INR


def cost_of_allowing_fraud(amount_inr: float) -> float:
    """False negative: fraud went through and will be charged back."""
    return amount_inr + CHARGEBACK_FEE_INR
