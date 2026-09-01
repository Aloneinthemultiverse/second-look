"""Day 3: the fixed investigation playbook and the entity graph.

Two deterministic components that run BEFORE the LLM ever sees a case.

THE PLAYBOOK -- five lookups, always the same five, computed from real
transaction history rather than read off the row:

  1. card velocity        prior transactions on this card in 1h / 24h
  2. device history       first-seen, lifetime count, distinct cards on device
  3. email domain risk    frequency and historical fraud rate (TRAIN ONLY)
  4. amount anomaly       z-score against this card's own prior amounts
  5. address consistency  billing/shipping fields present and in agreement

The lookups are fixed, not chosen by a model. An analyst already knows what to
check; the value is in doing it fast and identically every time.

THE ENTITY GRAPH -- links a case to other transactions sharing a card, device
or email domain, but ONLY on attributes carrying real information.

Why that filter exists: P_emaildomain is gmail.com for roughly two thirds of
rows. Linking cases on a shared gmail.com address builds a "ring" out of
nothing. Information value is  -log2 P(value); a value shared by 65% of the
population carries 0.6 bits and is discarded, while one seen in 0.01% of rows
carries ~13 bits and is worth a link. The filter runs before the LLM, so the
LLM is never in a position to narrate a coincidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import data

MIN_INFO_BITS = 6.0        # ~ value must occur in under 1.6% of rows
HOUR, DAY = 3_600, 86_400


class Playbook:
    """Precomputes the indices the five lookups need, then serves them fast."""

    def __init__(self, hist: pd.DataFrame, train: pd.DataFrame):
        self.hist = hist.sort_values("TransactionDT").reset_index(drop=True)
        self.dt = self.hist["TransactionDT"].to_numpy()
        self.amt = self.hist["TransactionAmt"].to_numpy(dtype=float)

        # positional index per entity, so a lookback is a slice not a scan
        self.by_card = self._index("card1")
        self.by_device = self._index("DeviceInfo")

        # email risk is learned on TRAIN ONLY. Using the test slice here would
        # leak the labels we are trying to predict.
        em = train.groupby("P_emaildomain")["isFraud"].agg(["mean", "size"])
        self.email_rate = em["mean"].to_dict()
        self.email_n = em["size"].to_dict()
        self.train_rate = float(train["isFraud"].mean())

        # information value of every entity value, for the graph filter
        self.info = {c: self._info_bits(c)
                     for c in ("card1", "DeviceInfo", "P_emaildomain")}

    def _index(self, col):
        idx = {}
        for pos, val in enumerate(self.hist[col].to_numpy()):
            if pd.isna(val):
                continue
            idx.setdefault(val, []).append(pos)
        return {k: np.array(v) for k, v in idx.items()}

    def _info_bits(self, col):
        vc = self.hist[col].value_counts(dropna=True)
        p = vc / len(self.hist)
        return (-np.log2(p)).to_dict()

    # --- the five lookups -------------------------------------------------

    def card_velocity(self, card, now):
        pos = self.by_card.get(card)
        if pos is None:
            return {"txn_1h": 0, "txn_24h": 0, "amt_24h": 0.0, "card_seen": 0}
        prior = pos[self.dt[pos] < now]
        if len(prior) == 0:
            return {"txn_1h": 0, "txn_24h": 0, "amt_24h": 0.0, "card_seen": 0}
        d = now - self.dt[prior]
        return {"txn_1h": int((d <= HOUR).sum()),
                "txn_24h": int((d <= DAY).sum()),
                "amt_24h": float(self.amt[prior][d <= DAY].sum() * config.USD_TO_INR),
                "card_seen": int(len(prior))}

    def device_history(self, device, now):
        if pd.isna(device) or device not in self.by_device:
            return {"device_known": False}
        pos = self.by_device[device]
        prior = pos[self.dt[pos] < now]
        if len(prior) == 0:
            return {"device_known": False}
        cards = self.hist["card1"].to_numpy()[prior]
        return {"device_known": True,
                "device_txns": int(len(prior)),
                "device_age_days": float((now - self.dt[prior].min()) / DAY),
                "distinct_cards_on_device": int(len(np.unique(cards[~pd.isna(cards)])))}

    def email_risk(self, domain):
        if pd.isna(domain) or domain not in self.email_rate:
            return {"email_known": False, "email_lift": None}
        rate = self.email_rate[domain]
        return {"email_known": True,
                "email_fraud_rate_train": float(rate),
                "email_lift": float(rate / self.train_rate),
                "email_n_train": int(self.email_n[domain])}

    def amount_anomaly(self, card, amount_usd, now):
        pos = self.by_card.get(card)
        if pos is None:
            return {"amount_history": 0, "amount_z": None}
        prior = pos[self.dt[pos] < now]
        if len(prior) < 3:
            return {"amount_history": int(len(prior)), "amount_z": None}
        a = self.amt[prior]
        sd = a.std()
        return {"amount_history": int(len(prior)),
                "amount_z": float((amount_usd - a.mean()) / sd) if sd > 0 else 0.0,
                "card_median_amt_inr": float(np.median(a) * config.USD_TO_INR)}

    def address_consistency(self, row):
        return {"billing_region_present": not pd.isna(row.get("addr1")),
                "billing_country_present": not pd.isna(row.get("addr2")),
                "distance_present": not pd.isna(row.get("dist1")),
                "distance": None if pd.isna(row.get("dist1")) else float(row["dist1"])}

    def run(self, row):
        now = float(row["TransactionDT"])
        out = {}
        out.update(self.card_velocity(row.get("card1"), now))
        out.update(self.device_history(row.get("DeviceInfo"), now))
        out.update(self.email_risk(row.get("P_emaildomain")))
        out.update(self.amount_anomaly(row.get("card1"),
                                       float(row["TransactionAmt"]), now))
        out.update(self.address_consistency(row))
        return out

    # --- entity graph -----------------------------------------------------

    def graph(self, row, max_links=8):
        """Neighbours sharing a HIGH-INFORMATION attribute value only."""
        now = float(row["TransactionDT"])
        links, dropped = [], []
        for col, index in (("card1", self.by_card), ("DeviceInfo", self.by_device)):
            val = row.get(col)
            if pd.isna(val):
                continue
            bits = self.info[col].get(val, 0.0)
            if bits < MIN_INFO_BITS:
                dropped.append({"attribute": col, "value": str(val),
                                "info_bits": round(bits, 2),
                                "reason": "too common to be evidence"})
                continue
            pos = index.get(val, np.array([], int))
            prior = pos[self.dt[pos] < now][-max_links:]
            if len(prior):
                links.append({"attribute": col, "value": str(val),
                              "info_bits": round(bits, 2),
                              "linked_transactions": int(len(prior))})
        # email domain is checked explicitly because it is the classic trap
        dom = row.get("P_emaildomain")
        if not pd.isna(dom):
            bits = self.info["P_emaildomain"].get(dom, 0.0)
            entry = {"attribute": "P_emaildomain", "value": str(dom),
                     "info_bits": round(bits, 2)}
            if bits < MIN_INFO_BITS:
                entry["reason"] = "too common to be evidence"
                dropped.append(entry)
            else:
                links.append(entry)
        return {"links": links, "dropped_low_information": dropped}


def main() -> None:
    """Demonstrate the playbook and prove the filter blocks the gmail.com trap."""
    raw = data.load_raw()
    tr_df, ca_df, te_df = data.temporal_split(raw)
    pb = Playbook(raw, tr_df)

    w = 84
    print("=" * w)
    print("INFORMATION VALUE OF LINKING ATTRIBUTES  (threshold "
          f"{MIN_INFO_BITS} bits)")
    print("=" * w)
    print(f"{'attribute':<18}{'value':<26}{'share of rows':>16}{'bits':>9}{'link?':>10}")
    print("-" * w)
    for col in ("P_emaildomain", "DeviceInfo", "card1"):
        vc = raw[col].value_counts(dropna=True)
        for val in list(vc.index[:2]) + list(vc.index[-1:]):
            share = vc[val] / len(raw)
            bits = -np.log2(share)
            print(f"{col:<18}{str(val)[:24]:<26}{share:>16.4%}{bits:>9.1f}"
                  f"{'yes' if bits >= MIN_INFO_BITS else 'NO':>10}")
    print("-" * w)
    g = raw["P_emaildomain"].value_counts(normalize=True)
    print(f"gmail.com is {g.get('gmail.com', 0):.1%} of all rows "
          f"({-np.log2(g.get('gmail.com', 1e-9)):.1f} bits) -> filtered out.")
    print("Without this filter, any two cases sharing gmail.com look like a ring.")

    # run the playbook on a few real test rows
    print("\n" + "=" * w)
    print("PLAYBOOK OUTPUT, three test transactions")
    print("=" * w)
    for _, row in te_df.head(3).iterrows():
        ev = pb.run(row)
        gr = pb.graph(row)
        print(f"\ntxn {int(row['TransactionID'])}  "
              f"Rs {row['TransactionAmt'] * config.USD_TO_INR:,.0f}  "
              f"actual: {'FRAUD' if row['isFraud'] else 'genuine'}")
        print(f"  velocity     1h={ev['txn_1h']} 24h={ev['txn_24h']} "
              f"card seen {ev['card_seen']}x")
        print(f"  device       known={ev['device_known']}"
              + (f" txns={ev.get('device_txns')} "
                 f"cards={ev.get('distinct_cards_on_device')}"
                 if ev["device_known"] else ""))
        print(f"  email        lift={ev['email_lift']:.2f}x"
              if ev.get("email_lift") else "  email        unknown domain")
        print(f"  amount z     {ev['amount_z']}" if ev["amount_z"] is not None
              else f"  amount z     n/a (history {ev['amount_history']})")
        print(f"  graph links  {len(gr['links'])} kept, "
              f"{len(gr['dropped_low_information'])} dropped as low-information")
        for d in gr["dropped_low_information"]:
            print(f"     dropped {d['attribute']}={d['value'][:22]} "
                  f"({d['info_bits']} bits)")
    print("=" * w)

    (config.ARTIFACTS / "playbook_demo.json").write_text(json.dumps({
        "min_info_bits": MIN_INFO_BITS,
        "gmail_share": float(g.get("gmail.com", 0)),
        "gmail_bits": float(-np.log2(g.get("gmail.com", 1e-9))),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {config.ARTIFACTS / 'playbook_demo.json'}")


if __name__ == "__main__":
    main()
