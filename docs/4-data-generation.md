Yes 👍 let’s build this properly.

I’ll give you:

1. ✅ A **clean generator architecture**
2. ✅ Realistic financial-style templates
3. ✅ Noise injection (invoice numbers, dates, vendors)
4. ✅ Scalable to all 77 DPL tags
5. ✅ Ready-to-run Python script

We’ll design it so you can easily extend it.

---

# 🧠 Design Principles for Good Synthetic Data

To make it realistic:

* Mix structured + messy text
* Include vendor names
* Include invoice numbers
* Add departments
* Add currencies
* Vary formatting
* Include accounting phrasing

Example realistic outputs:

```
INV-78451 – Deloitte LLP – Audit services FY2025
Interest charged on Barclays overdraft – Mar 2026
Foreign exchange loss on USD settlement – AP 8834
AWS cloud hosting – IT dept – Feb subscription
Office rent – London HQ – April 2026
```

---

# 🏗 FULL SYNTHETIC DATA GENERATOR

Below is a scalable version.
You can later extend templates for all tags.

---

## 🔹 Step 1 — Base Generator Class

```python
import random
import pandas as pd
from datetime import datetime

class SyntheticDPLGenerator:

    def __init__(self):

        self.months = [
            "January", "February", "March", "April", "May",
            "June", "July", "August", "September", "October",
            "November", "December"
        ]

        self.years = ["2024", "2025", "2026"]

        self.currencies = ["USD", "GBP", "EUR"]

        self.departments = [
            "Finance", "HR", "IT", "Operations", "Marketing",
            "Legal", "Procurement"
        ]

        self.bank_names = ["HSBC", "Barclays", "Citibank", "Standard Chartered"]

        self.audit_firms = ["Deloitte", "PwC", "KPMG", "EY"]

        self.law_firms = ["Baker McKenzie", "Clifford Chance", "Linklaters"]

        self.it_vendors = ["AWS", "Microsoft Azure", "Google Cloud"]

        self.general_vendors = ["ABC Ltd", "Global Services Inc", "Prime Solutions"]

        self.templates = self._build_templates()

    def _random_invoice(self):
        return f"INV-{random.randint(10000,99999)}"

    def _random_month(self):
        return random.choice(self.months)

    def _random_year(self):
        return random.choice(self.years)

    def _random_currency_amount(self):
        return f"{random.choice(self.currencies)} {random.randint(1000,50000)}"

    def _build_templates(self):
        return {

            # -----------------------
            # DPL003 – Audit & Tax
            # -----------------------
            "DPL003": [
                lambda: f"{self._random_invoice()} – {random.choice(self.audit_firms)} audit services FY{self._random_year()}",
                lambda: f"Tax advisory fees – {random.choice(self.audit_firms)}",
                lambda: f"External audit fee for year ended {self._random_year()}"
            ],

            # -----------------------
            # DPL035 – Bank Interest Expense
            # -----------------------
            "DPL035": [
                lambda: f"Interest charged on {random.choice(self.bank_names)} overdraft – {self._random_month()}",
                lambda: f"Loan interest expense – term facility – {self._random_year()}",
                lambda: f"Bank interest debit – revolving credit facility"
            ],

            # -----------------------
            # DPL006 – Bank Interest Income
            # -----------------------
            "DPL006": [
                lambda: f"Interest income received from {random.choice(self.bank_names)} savings account",
                lambda: f"Bank interest credited – {self._random_month()}",
                lambda: f"Interest received on cash deposits"
            ],

            # -----------------------
            # DPL038 – Legal
            # -----------------------
            "DPL038": [
                lambda: f"{self._random_invoice()} – Legal advisory fees – {random.choice(self.law_firms)}",
                lambda: f"Litigation expense – contract dispute",
                lambda: f"Corporate legal services – acquisition support"
            ],

            # -----------------------
            # DPL037 – IT Costs
            # -----------------------
            "DPL037": [
                lambda: f"{random.choice(self.it_vendors)} cloud hosting – {self._random_month()} subscription",
                lambda: f"Software license renewal – IT department",
                lambda: f"Annual ERP maintenance fee"
            ],

            # -----------------------
            # DPL055 – Rent
            # -----------------------
            "DPL055": [
                lambda: f"Office rent – HQ building – {self._random_month()} {self._random_year()}",
                lambda: f"Warehouse lease payment",
                lambda: f"Property rental expense – regional office"
            ],

            # -----------------------
            # DPL068 – Wages
            # -----------------------
            "DPL068": [
                lambda: f"Monthly payroll – {self._random_month()} {self._random_year()}",
                lambda: f"Salaries and wages expense",
                lambda: f"Employee payroll processing"
            ],

            # -----------------------
            # DPL065 – Travel
            # -----------------------
            "DPL065": [
                lambda: f"Flight booking – business trip – {self._random_month()}",
                lambda: f"Hotel accommodation – client visit",
                lambda: f"Travel reimbursement – employee expense claim"
            ],

            # -----------------------
            # DPL045 – Other Operating Income
            # -----------------------
            "DPL045": [
                lambda: f"Other operating income – service recharge",
                lambda: f"Miscellaneous operating income received",
                lambda: f"Operational income adjustment"
            ],

            # -----------------------
            # DPL074 – Other Costs
            # -----------------------
            "DPL074": [
                lambda: f"Miscellaneous expense",
                lambda: f"General cost allocation",
                lambda: f"Other cost – departmental adjustment"
            ],
        }

    def generate_for_tag(self, tag, n=300):
        records = []

        if tag not in self.templates:
            return []

        for _ in range(n):
            template = random.choice(self.templates[tag])
            description = template()

            records.append({
                "description": description,
                "dpl_tag": tag
            })

        return records

    def generate_dataset(self, tags, n_per_tag=300):
        all_data = []

        for tag in tags:
            all_data.extend(self.generate_for_tag(tag, n_per_tag))

        return pd.DataFrame(all_data)
```

---

# 🔹 Step 2 — Generate Dataset

```python
tags = [
    "DPL003",
    "DPL035",
    "DPL006",
    "DPL038",
    "DPL037",
    "DPL055",
    "DPL068",
    "DPL065",
    "DPL045",
    "DPL074"
]

generator = SyntheticDPLGenerator()
df = generator.generate_dataset(tags, n_per_tag=300)

df.to_csv("synthetic_dpl_dataset.csv", index=False)

print(df.sample(10))
```

---

# 📊 Output Example

| description                                | dpl_tag |
| ------------------------------------------ | ------- |
| INV-78345 – Deloitte audit services FY2025 | DPL003  |
| Interest charged on HSBC overdraft – March | DPL035  |
| Monthly payroll – April 2026               | DPL068  |
| AWS cloud hosting – February subscription  | DPL037  |

---

# 🚀 How To Scale To All 77 Tags

You just:

1. Add more template blocks inside `_build_templates()`
2. Maintain 5–10 templates per tag
3. Add variation keywords

---

# ⚠️ VERY IMPORTANT

Synthetic data must:

* NOT be too clean
* Include ambiguous wording
* Include overlapping phrases
* Include noise

Otherwise your model will overfit and fail in real data.

---

If you want next, I can:

* ✅ Expand this to cover all 77 tags
* ✅ Add automatic confusion-stress generator
* ✅ Add realistic ERP noise injection
* ✅ Convert this into HuggingFace dataset pipeline
* ✅ Build hierarchical training code

Tell me what level you want next 🚀
