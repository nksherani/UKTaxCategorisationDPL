"""
Synthetic DPL Training Data Generator
Covers all 76 active DPL tags (DPL001–DPL076, skipping DPL000 which is reserved).

Design principles:
  - Mix structured and messy text (vendor names, invoice refs, dates, amounts)
  - Include ambiguous and overlapping phrasing to mirror real ERP data
  - 15–30 template lambdas per tag for variety (expanded from original 5–10)
  - Noise injection via randomised helpers
  - Post-generation deduplication — no exact duplicates in final dataset

Usage:
    python generate_dpl_data.py                  # generates datasets/ folder with train/val/test CSVs
    python generate_dpl_data.py --n 500          # 500 unique samples per tag (duplicates removed)
    python generate_dpl_data.py --out my_dir     # write to my_dir/
"""

import random
import argparse
import os
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class SyntheticDPLGenerator:

    def __init__(self):
        self.months = [
            "January", "February", "March", "April", "May",
            "June", "July", "August", "September", "October",
            "November", "December",
            "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec",
        ]
        self.quarters = ["Q1", "Q2", "Q3", "Q4"]
        self.years = ["2023", "2024", "2025", "2026"]
        self.currencies = ["USD", "GBP", "EUR", "CHF", "JPY"]
        self.departments = [
            "Finance", "HR", "IT", "Operations", "Marketing",
            "Legal", "Procurement", "Sales", "R&D", "Facilities",
        ]
        self.bank_names = [
            "HSBC", "Barclays", "Citibank", "Standard Chartered",
            "Lloyds", "NatWest", "Santander", "Deutsche Bank",
            "RBS", "Halifax", "Nationwide", "Metro Bank",
            "JP Morgan", "Bank of America", "Wells Fargo",
        ]
        self.audit_firms = [
            "Deloitte", "PwC", "KPMG", "EY", "BDO",
            "Grant Thornton", "Mazars", "RSM", "Crowe", "Moore Kingston Smith",
        ]
        self.law_firms = [
            "Baker McKenzie", "Clifford Chance", "Linklaters",
            "Freshfields", "Allen & Overy", "Herbert Smith Freehills",
            "Slaughter and May", "DLA Piper", "Norton Rose Fulbright",
            "CMS", "Eversheds Sutherland", "Ashurst",
        ]
        self.it_vendors = [
            "AWS", "Microsoft Azure", "Google Cloud", "Oracle",
            "SAP", "Salesforce", "ServiceNow", "Workday",
            "IBM", "VMware", "Cisco", "Dell Technologies",
            "Snowflake", "Databricks", "HashiCorp",
        ]
        self.telecom_vendors = [
            "BT", "Vodafone", "EE", "O2", "Virgin Media",
            "Gamma", "Talk Talk", "Three", "Sky Business",
            "Verizon", "AT&T", "Twilio",
        ]
        self.logistics_vendors = [
            "DHL", "FedEx", "UPS", "TNT", "Hermes", "Yodel", "Wincanton",
            "XPO Logistics", "DB Schenker", "Kuehne+Nagel",
            "Ceva Logistics", "Geodis", "Agility",
        ]
        self.insurance_vendors = [
            "Aviva", "AXA", "Zurich", "Allianz", "Hiscox", "RSA",
            "QBE", "Chubb", "Liberty Mutual", "Travelers",
            "Markel", "Beazley",
        ]
        self.general_vendors = [
            "ABC Ltd", "Global Services Inc", "Prime Solutions",
            "Nexus Group", "Apex Services", "Summit Consulting",
            "Horizon Partners", "Blue Ridge Corp", "Sterling Group",
            "Crestwood Ltd", "Elmwood Associates", "Redwood Partners",
            "Lakeside Corp", "Pinewood Ltd", "Northfield Services",
        ]
        self.charities = [
            "Red Cross", "Cancer Research UK", "British Heart Foundation",
            "Oxfam", "NSPCC", "Macmillan Cancer Support",
            "Save the Children", "Age UK", "Shelter", "Mind",
            "WWF", "Comic Relief", "Barnardo's", "Action Aid",
        ]
        self.clients = [
            "Acme Corp", "Pinnacle Group", "Zenith Ltd",
            "Titan Industries", "Atlas Holdings", "Meridian PLC",
            "Vanguard Ltd", "Sterling PLC", "Apex Group",
            "Corinthian Ltd", "Halcyon Partners", "Solaris Inc",
        ]
        self.locations = [
            "London HQ", "Manchester office", "Edinburgh branch",
            "Birmingham site", "Leeds depot", "Bristol office",
            "Glasgow office", "Cardiff branch", "Sheffield site",
            "Liverpool depot", "Nottingham office", "Reading HQ",
        ]
        self.subsidiaries = [
            "Subsidiary A", "Northern Division", "Southern Operations",
            "NewCo Ltd", "Holdco Ltd", "JV Entity",
            "Eastern Region Ltd", "Western Trading Ltd", "Asia Pacific Holdco",
            "European Division", "US Operations Inc", "LatAm Subsidiary",
        ]
        self.projects = [
            "Phoenix", "Horizon", "Titan", "Project Alpha",
            "Orion", "Vertex", "Nexus", "Aurora",
            "Sapphire", "Cobalt", "Project Delta", "Project Zeta",
            "Falcon", "Mercury", "Apollo",
        ]
        self.ip_names = [
            "brand logo", "software platform", "patented process",
            "trade mark portfolio", "licensed technology",
            "proprietary algorithm", "registered design", "database rights",
            "copyright content", "trade secret",
        ]
        self.ip_holders = [
            "Patent Holder Ltd", "IP Licensor Inc", "Rights Management Co",
            "TechIP Group", "Brand Owners PLC",
            "Global IP Holdings", "Creative Rights Ltd", "Innovation Licensing Corp",
            "Portfolio IP Ltd", "IP Asset Management Ltd",
        ]
        self.grant_names = [
            "Innovate UK", "Enterprise Development Grant",
            "Regional Growth Fund", "Export Development Grant",
            "Business Investment Grant", "Catapult Innovation Fund",
            "UKRI grant", "Horizon Europe funding",
            "Levelling Up Fund", "UK Shared Prosperity Fund",
            "Made Smarter grant", "KTP award",
        ]
        self.subscription_vendors = [
            "LinkedIn", "Bloomberg", "Reuters", "Lexis Nexis",
            "Westlaw", "Sage", "Xero", "QuickBooks",
            "Refinitiv", "S&P Capital IQ", "Moody's Analytics",
            "PitchBook", "Mergermarket", "Dun & Bradstreet",
        ]
        self.regulators = [
            "HMRC", "FCA", "CMA", "ICO", "Environment Agency",
            "HSE", "Ofgem", "Ofcom", "PRA", "SRA",
            "Companies House", "Pensions Regulator", "CQC",
        ]
        self.parent_companies = [
            "Parent Co PLC", "Group HQ Ltd", "Holdco Group",
            "Ultimate Parent Corp", "Group Services Ltd",
            "Group Finance Ltd", "Corporate Centre Ltd",
            "Shared Services Co", "Treasury Centre Ltd",
        ]
        # Additional pools for specific tags
        self.political_parties = [
            "Conservative Party", "Labour Party", "Liberal Democrats",
            "SNP", "Green Party", "Reform UK",
        ]
        self.instruments = [
            "interest rate swap", "cross-currency swap", "FX forward contract",
            "equity option", "commodity futures", "credit default swap",
            "inflation swap", "total return swap", "currency option",
        ]
        self.asset_classes = [
            "listed equities", "corporate bonds", "government gilts",
            "private equity fund", "hedge fund", "REIT",
            "infrastructure fund", "venture capital trust",
        ]
        self.provisions = [
            "restructuring provision", "dilapidation provision",
            "warranty provision", "onerous contract provision",
            "decommissioning provision", "legal claims provision",
            "environmental remediation provision",
        ]
        self.currencies_full = [
            "USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD",
            "SGD", "HKD", "NOK", "SEK", "DKK",
        ]

        self.templates = self._build_templates()

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------

    def _inv(self):
        """Random invoice / reference number."""
        prefix = random.choice(["INV", "AP", "PO", "REF", "JNL", "EXP"])
        return f"{prefix}-{random.randint(10000, 99999)}"

    def _month(self):
        return random.choice(self.months)

    def _quarter(self):
        return random.choice(self.quarters)

    def _year(self):
        return random.choice(self.years)

    def _amt(self):
        ccy = random.choice(self.currencies)
        val = random.choice([
            random.randint(500, 9999),
            random.randint(10000, 99999),
            random.randint(100000, 999999),
        ])
        return f"{ccy} {val:,}"

    def _dept(self):
        return random.choice(self.departments)

    def _bank(self):
        return random.choice(self.bank_names)

    def _firm(self):
        return random.choice(self.audit_firms)

    def _law(self):
        return random.choice(self.law_firms)

    def _it(self):
        return random.choice(self.it_vendors)

    def _tel(self):
        return random.choice(self.telecom_vendors)

    def _log(self):
        return random.choice(self.logistics_vendors)

    def _ins(self):
        return random.choice(self.insurance_vendors)

    def _gen(self):
        return random.choice(self.general_vendors)

    def _charity(self):
        return random.choice(self.charities)

    def _client(self):
        return random.choice(self.clients)

    def _loc(self):
        return random.choice(self.locations)

    def _sub(self):
        return random.choice(self.subsidiaries)

    def _proj(self):
        return random.choice(self.projects)

    def _ip(self):
        return random.choice(self.ip_names)

    def _iph(self):
        return random.choice(self.ip_holders)

    def _grant(self):
        return random.choice(self.grant_names)

    def _subv(self):
        return random.choice(self.subscription_vendors)

    def _reg(self):
        return random.choice(self.regulators)

    def _parent(self):
        return random.choice(self.parent_companies)

    def _party(self):
        return random.choice(self.political_parties)

    def _instr(self):
        return random.choice(self.instruments)

    def _asset(self):
        return random.choice(self.asset_classes)

    def _prov(self):
        return random.choice(self.provisions)

    def _ccy(self):
        return random.choice(self.currencies_full)

    # ------------------------------------------------------------------
    # Template bank
    # ------------------------------------------------------------------

    def _build_templates(self):  # noqa: C901 (long but intentional)
        t = self

        return {

            # ----------------------------------------------------------------
            # DPL001 – Advertising, promotions and marketing costs
            # ----------------------------------------------------------------
            "DPL001": [
                lambda: f"Google Ads campaign – {t._quarter()} {t._year()} – {t._dept()}",
                lambda: f"{t._inv()} – Digital marketing services",
                lambda: f"Social media advertising spend – {t._month()} {t._year()}",
                lambda: f"TV and radio campaign production costs – {t._year()}",
                lambda: f"Marketing agency retainer – {t._month()}",
                lambda: f"Trade show and exhibition expenses – {t._year()}",
                lambda: f"Brand promotion campaign – {t._gen()}",
                lambda: f"PR agency fees – {t._month()} {t._year()}",
                lambda: f"Promotional materials and giveaways – {t._quarter()}",
                lambda: f"Influencer marketing costs – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL002 – Amortisation expense, intangible assets
            # ----------------------------------------------------------------
            "DPL002": [
                lambda: f"Amortisation of software licence – {t._month()} {t._year()}",
                lambda: f"Patent amortisation charge for period ending {t._year()}",
                lambda: f"Amortisation – customer relationship intangible – {t._year()}",
                lambda: f"Software development costs amortised – {t._month()} {t._year()}",
                lambda: f"Amortisation charge – brand name – {t._year()}",
                lambda: f"Intangible asset amortisation – {t._dept()} system – {t._year()}",
                lambda: f"Amortisation of acquired technology – {t._sub()} – {t._year()}",
                lambda: f"Goodwill amortisation – legacy policy – {t._year()}",
                lambda: f"Amortisation – {t._proj()} software – {t._month()} {t._year()}",
                lambda: f"Customer list amortisation – {t._year()}",
                lambda: f"Amortisation of franchise agreement – {t._year()}",
                lambda: f"Non-compete agreement amortisation – {t._year()}",
                lambda: f"Amortisation – acquired order backlog – {t._year()}",
                lambda: f"Technology intangible amortisation – {t._year()}",
                lambda: f"Amortisation charge – {t._inv()} – intangible assets – {t._year()}",
                lambda: f"ERP system amortisation – {t._it()} – {t._month()} {t._year()}",
                lambda: f"Licence amortisation – {t._it()} platform – {t._year()}",
                lambda: f"Domain name amortisation – {t._year()}",
                lambda: f"Subscriber list amortisation – {t._year()}",
                lambda: f"Amortisation – distribution rights – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL003 – Audit and accountancy, tax services
            # ----------------------------------------------------------------
            "DPL003": [
                lambda: f"{t._inv()} – {t._firm()} audit services FY{t._year()}",
                lambda: f"Tax advisory fees – {t._firm()}",
                lambda: f"External audit fee for year ended {t._year()}",
                lambda: f"Corporation tax return preparation fees – {t._firm()}",
                lambda: f"VAT compliance services – {t._month()} {t._year()}",
                lambda: f"Accountancy fees – year end accounts – {t._year()}",
                lambda: f"Transfer pricing report – {t._firm()}",
                lambda: f"Tax due diligence – {t._firm()} – {t._year()}",
                lambda: f"Audit planning fees – {t._firm()} – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL004 – Bad debts and impairment losses
            # ----------------------------------------------------------------
            "DPL004": [
                lambda: f"Bad debt write-off – customer {t._inv()}",
                lambda: f"Trade receivable impairment charge – {t._month()} {t._year()}",
                lambda: f"Provision for bad debts – increase – {t._year()}",
                lambda: f"Debt impairment loss – {t._gen()}",
                lambda: f"Write-off of irrecoverable debt – {t._amt()}",
                lambda: f"Expected credit loss provision – {t._month()}",
                lambda: f"Doubtful debt provision charge – {t._year()}",
                lambda: f"Bad debt expense – overdue receivable {t._inv()}",
            ],

            # ----------------------------------------------------------------
            # DPL005 – Bank charges
            # ----------------------------------------------------------------
            "DPL005": [
                lambda: f"{t._bank()} account maintenance fee – {t._month()}",
                lambda: f"Bank service charges – {t._month()} {t._year()}",
                lambda: f"Transaction processing fees – {t._bank()}",
                lambda: f"SWIFT transfer charges – {t._month()}",
                lambda: f"Bank arrangement and facility fees",
                lambda: f"Current account charges – {t._bank()} – {t._month()}",
                lambda: f"Foreign currency conversion fee – {t._bank()}",
                lambda: f"Bank administration fee – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL006 – Bank interest and similar income receivable
            # ----------------------------------------------------------------
            "DPL006": [
                lambda: f"Interest income received from {t._bank()} savings account",
                lambda: f"Bank interest credited – {t._month()} {t._year()}",
                lambda: f"Interest received on cash deposits – {t._year()}",
                lambda: f"{t._bank()} current account interest income – {t._month()}",
                lambda: f"Deposit account interest income – {t._year()}",
                lambda: f"Interest on term deposit – {t._bank()} – {t._month()}",
                lambda: f"Short-term investment interest received",
                lambda: f"Treasury deposit interest – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL007 – Charitable donations
            # ----------------------------------------------------------------
            "DPL007": [
                lambda: f"Charitable donation – {t._charity()}",
                lambda: f"Corporate social responsibility donation – {t._month()} {t._year()}",
                lambda: f"Gift aid donation to {t._charity()}",
                lambda: f"Community foundation contribution – {t._year()}",
                lambda: f"Charitable giving – year end {t._year()}",
                lambda: f"Employee matched giving – {t._charity()}",
                lambda: f"Fundraising event costs – {t._charity()} – {t._month()}",
                lambda: f"Donation to {t._charity()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL008 – Cleaning costs
            # ----------------------------------------------------------------
            "DPL008": [
                lambda: f"Office cleaning services – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Cleaning contractor – {t._gen()}",
                lambda: f"Facilities cleaning – {t._loc()} – {t._month()}",
                lambda: f"Building maintenance cleaning charge",
                lambda: f"Window cleaning – quarterly service – {t._year()}",
                lambda: f"Deep clean – {t._loc()} – {t._month()}",
                lambda: f"Janitorial services – {t._month()} {t._year()}",
                lambda: f"Cleaning supplies and services – {t._dept()}",
            ],

            # ----------------------------------------------------------------
            # DPL009 – Consultancy costs
            # ----------------------------------------------------------------
            "DPL009": [
                lambda: f"Management consultancy fees – {t._inv()}",
                lambda: f"{t._gen()} – Strategy consulting services – {t._month()} {t._year()}",
                lambda: f"Business transformation consulting – {t._month()} {t._year()}",
                lambda: f"IT consultancy project fees – {t._year()}",
                lambda: f"{t._firm()} management consulting – {t._proj()} project",
                lambda: f"HR consultancy – restructuring project – {t._year()}",
                lambda: f"Finance transformation advisory – {t._inv()}",
                lambda: f"Operational consultancy – {t._gen()} – {t._month()}",
                lambda: f"Technical consulting services – {t._it()} implementation",
            ],

            # ----------------------------------------------------------------
            # DPL010 – Depreciation expense, property, plant and equipment
            # ----------------------------------------------------------------
            "DPL010": [
                lambda: f"Monthly depreciation charge – plant and machinery – {t._month()} {t._year()}",
                lambda: f"Depreciation – office furniture and equipment – {t._month()} {t._year()}",
                lambda: f"PPE depreciation charge for period",
                lambda: f"Vehicle fleet depreciation – {t._year()}",
                lambda: f"Building improvements depreciation – {t._year()}",
                lambda: f"Leasehold improvements depreciation charge – {t._month()}",
                lambda: f"Machinery depreciation – {t._dept()} – {t._month()}",
                lambda: f"Straight-line depreciation – fixed assets – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL011 – Environmental costs
            # ----------------------------------------------------------------
            "DPL011": [
                lambda: f"Environmental compliance costs – {t._year()}",
                lambda: f"Carbon offset purchase – {t._month()} {t._year()}",
                lambda: f"Environmental monitoring and reporting expense – {t._year()}",
                lambda: f"Waste disposal and environmental fees – {t._month()} {t._year()}",
                lambda: f"Pollution remediation costs – {t._loc()}",
                lambda: f"ETS carbon allowance purchase – {t._year()}",
                lambda: f"Environmental audit costs – {t._gen()} – {t._year()}",
                lambda: f"Emissions monitoring service – {t._year()}",
                lambda: f"Carbon credit purchase – {t._month()} {t._year()}",
                lambda: f"Greenhouse gas reporting costs – {t._year()}",
                lambda: f"Environmental permit fees – {t._reg()} – {t._year()}",
                lambda: f"Waste management and recycling costs – {t._loc()} – {t._year()}",
                lambda: f"Environmental impact assessment – {t._gen()} – {t._year()}",
                lambda: f"Net zero carbon strategy costs – {t._year()}",
                lambda: f"SECR energy and carbon reporting – {t._year()}",
                lambda: f"Contaminated land remediation – {t._loc()} – {t._year()}",
                lambda: f"Environmental consultancy – {t._gen()} – {t._year()}",
                lambda: f"Biodiversity net gain costs – {t._year()}",
                lambda: f"Carbon footprint assessment – {t._year()}",
                lambda: f"Scope 1 and 2 emissions audit costs – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL012 – External commission costs
            # ----------------------------------------------------------------
            "DPL012": [
                lambda: f"Sales commission – external agent – {t._month()} {t._year()}",
                lambda: f"Referral commission paid – {t._inv()}",
                lambda: f"External distribution commission – {t._gen()}",
                lambda: f"Broker commission expense – {t._month()}",
                lambda: f"Agent commission – {t._month()} {t._year()}",
                lambda: f"Introducer fee – {t._gen()} – {t._year()}",
                lambda: f"Third party sales commission – {t._inv()}",
                lambda: f"Channel partner commission – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL013 – Financial instrument fair value GAIN
            # ----------------------------------------------------------------
            "DPL013": [
                lambda: f"Fair value GAIN on {t._instr()} – {t._month()} {t._year()}",
                lambda: f"Derivative financial instrument fair value UPLIFT – {t._year()}",
                lambda: f"Mark-to-market GAIN – {t._instr()} – {t._month()}",
                lambda: f"Investment securities fair value GAIN – {t._year()}",
                lambda: f"Equity derivative fair value GAIN – {t._month()}",
                lambda: f"Financial asset FVTPL fair value INCREASE – {t._year()}",
                lambda: f"Unrealised GAIN on {t._instr()} – {t._month()} {t._year()}",
                lambda: f"Positive mark-to-market movement – {t._instr()} – {t._year()}",
                lambda: f"Derivative asset fair value UPLIFT – {t._month()} {t._year()}",
                lambda: f"FVTPL asset revaluation GAIN – {t._asset()} – {t._year()}",
                lambda: f"Financial instrument MTM GAIN – {t._instr()} – {t._month()}",
                lambda: f"Hedging instrument GAIN – {t._instr()} – {t._year()}",
                lambda: f"Realised fair value GAIN on {t._instr()} – {t._year()}",
                lambda: f"Investment portfolio FVTPL GAIN – {t._month()} {t._year()}",
                lambda: f"{t._asset()} fair value INCREASE – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL014 – Financial instrument fair value LOSS
            # ----------------------------------------------------------------
            "DPL014": [
                lambda: f"Fair value LOSS on {t._instr()} – {t._month()} {t._year()}",
                lambda: f"Derivative financial instrument fair value DECLINE – {t._year()}",
                lambda: f"Mark-to-market LOSS – {t._instr()} – {t._month()}",
                lambda: f"Investment securities fair value LOSS – {t._year()}",
                lambda: f"Equity derivative mark-to-market LOSS – {t._month()}",
                lambda: f"Financial liability FVTPL fair value INCREASE – {t._year()}",
                lambda: f"Unrealised LOSS on {t._instr()} – {t._month()} {t._year()}",
                lambda: f"Negative mark-to-market movement – {t._instr()} – {t._year()}",
                lambda: f"Derivative liability fair value CHARGE – {t._month()} {t._year()}",
                lambda: f"FVTPL asset revaluation LOSS – {t._asset()} – {t._year()}",
                lambda: f"Financial instrument MTM LOSS – {t._instr()} – {t._month()}",
                lambda: f"Hedging instrument LOSS – {t._instr()} – {t._year()}",
                lambda: f"Realised fair value LOSS on {t._instr()} – {t._year()}",
                lambda: f"Investment portfolio FVTPL LOSS – {t._month()} {t._year()}",
                lambda: f"{t._asset()} fair value DECREASE – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL015 – Fines and penalties
            # ----------------------------------------------------------------
            "DPL015": [
                lambda: f"{t._reg()} late filing penalty – {t._year()}",
                lambda: f"Regulatory fine – {t._reg()} – {t._year()}",
                lambda: f"Contract penalty charge – {t._inv()}",
                lambda: f"Compliance breach fine – {t._year()}",
                lambda: f"GDPR penalty – data breach – {t._year()}",
                lambda: f"Environmental penalty – {t._reg()} – {t._month()}",
                lambda: f"Late payment penalty – {t._inv()}",
                lambda: f"Customs penalty and surcharge – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL016 – Freight and haulage costs
            # ----------------------------------------------------------------
            "DPL016": [
                lambda: f"Haulage costs – goods delivery – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Freight charges – {t._log()}",
                lambda: f"Outbound shipping costs – {t._month()}",
                lambda: f"Import freight costs – {t._amt()}",
                lambda: f"Logistics and distribution expenses – {t._year()}",
                lambda: f"Road haulage – {t._log()} – {t._month()}",
                lambda: f"Air freight charges – {t._inv()} – {t._month()}",
                lambda: f"Sea freight – container shipment – {t._year()}",
                lambda: f"Courier and express delivery – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL017 – Gain / (loss) after tax on sale of discontinued operations
            # ----------------------------------------------------------------
            "DPL017": [
                lambda: f"Gain on disposal of {t._sub()} – discontinued operations – {t._year()}",
                lambda: f"Loss on discontinued operations – {t._sub()} – {t._year()}",
                lambda: f"Post-tax gain on sale of {t._sub()} division – {t._year()}",
                lambda: f"Discontinued segment disposal gain/(loss) – {t._year()}",
                lambda: f"Net gain on derecognition of discontinued business – {t._year()}",
                lambda: f"Disposal of {t._sub()} – after-tax gain – {t._year()}",
                lambda: f"IFRS 5 discontinued operations gain – {t._sub()} – {t._year()}",
                lambda: f"After-tax loss on sale of {t._sub()} – {t._year()}",
                lambda: f"Discontinued operations – net gain after tax – {t._year()}",
                lambda: f"Disposal proceeds less carrying value – {t._sub()} – {t._year()}",
                lambda: f"Sale of {t._sub()} business unit – post-tax gain – {t._year()}",
                lambda: f"Discontinued division disposal loss – {t._year()}",
                lambda: f"IFRS 5 held-for-sale disposal gain – {t._sub()} – {t._year()}",
                lambda: f"Net profit on disposal – {t._sub()} – discontinued – {t._year()}",
                lambda: f"Post-tax result on discontinued segment – {t._year()}",
                lambda: f"Gain on derecognition – discontinued operations – {t._sub()} – {t._year()}",
                lambda: f"Business disposal gain after income tax – {t._year()}",
                lambda: f"Loss on exit of {t._sub()} business – post-tax – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL018 – Gain / (loss) from changes in provisions
            # ----------------------------------------------------------------
            "DPL018": [
                lambda: f"Release of restructuring provision – {t._month()} {t._year()}",
                lambda: f"Provision movement – warranty claims – {t._year()}",
                lambda: f"Increase in dilapidation provision – {t._loc()}",
                lambda: f"Provision reversal – litigation settlement – {t._year()}",
                lambda: f"Onerous contract provision change – {t._month()}",
                lambda: f"Decommissioning provision re-measurement – {t._year()}",
                lambda: f"Legal provision released – {t._inv()}",
                lambda: f"Warranty provision utilisation – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL019 – Gain / (loss) from fair value adjustment, biological assets
            # ----------------------------------------------------------------
            "DPL019": [
                lambda: f"Fair value adjustment – livestock biological assets – {t._year()}",
                lambda: f"Agricultural produce fair value gain – harvest {t._year()}",
                lambda: f"Biological assets revaluation loss – {t._year()}",
                lambda: f"Poultry stock fair value adjustment – {t._month()}",
                lambda: f"Crop biological asset fair value gain – {t._year()}",
                lambda: f"IAS 41 biological asset measurement – {t._year()}",
                lambda: f"Cattle herd fair value uplift – {t._year()}",
                lambda: f"Dairy herd biological asset revaluation – {t._month()} {t._year()}",
                lambda: f"Sheep flock fair value loss – {t._year()}",
                lambda: f"Timber plantation biological asset gain – {t._year()}",
                lambda: f"Aquaculture fish stock fair value adjustment – {t._year()}",
                lambda: f"Vineyard grape crop fair value gain – harvest {t._year()}",
                lambda: f"IAS 41 bearer plant fair value – {t._month()} {t._year()}",
                lambda: f"Pig herd revaluation gain – {t._year()}",
                lambda: f"Fruit orchard biological asset – fair value movement – {t._year()}",
                lambda: f"Wool-producing flock IAS 41 adjustment – {t._year()}",
                lambda: f"Broiler chicken stock fair value – {t._month()}",
                lambda: f"Salmon farm biological asset gain/(loss) – {t._year()}",
                lambda: f"Agricultural biological assets – annual fair value movement – {t._year()}",
                lambda: f"Farm livestock revaluation – IAS 41 – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL020 – Gain / (loss) from fair value adjustment, investment property
            # ----------------------------------------------------------------
            "DPL020": [
                lambda: f"Investment property revaluation gain – {t._month()} {t._year()}",
                lambda: f"Fair value loss on investment property – {t._year()}",
                lambda: f"Property portfolio fair value adjustment – {t._year()}",
                lambda: f"Commercial property revaluation gain – {t._loc()}",
                lambda: f"IAS 40 investment property uplift – {t._year()}",
                lambda: f"Residential investment property fair value loss – {t._year()}",
                lambda: f"Property fund valuation gain – {t._month()} {t._year()}",
                lambda: f"Investment property fair value movement – {t._loc()} – {t._year()}",
                lambda: f"IAS 40 fair value gain on property portfolio – {t._year()}",
                lambda: f"Rental property revaluation – {t._loc()} – {t._year()}",
                lambda: f"Investment property uplift – {t._inv()} – {t._year()}",
                lambda: f"Office block investment property revaluation – {t._year()}",
                lambda: f"Retail property fair value gain/(loss) – {t._year()}",
                lambda: f"Investment property impairment reversed – {t._year()}",
                lambda: f"IAS 40 annual fair value update – {t._loc()} – {t._year()}",
                lambda: f"Property investment revaluation surplus – {t._year()}",
                lambda: f"Fair value movement – investment property – {t._year()}",
                lambda: f"Valuer assessment – investment property gain – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL021 – Gain / (loss) on disposals of intangible assets
            # ----------------------------------------------------------------
            "DPL021": [
                lambda: f"Gain on disposal of software licence – {t._year()}",
                lambda: f"Loss on disposal of trademark – {t._year()}",
                lambda: f"Intangible asset disposal gain – {t._inv()}",
                lambda: f"Patent disposal proceeds – net gain – {t._year()}",
                lambda: f"Customer list disposal – gain/(loss) – {t._year()}",
                lambda: f"Domain name sale proceeds – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL022 – Gain / (loss) on disposals of property, plant and equipment
            # ----------------------------------------------------------------
            "DPL022": [
                lambda: f"Profit on disposal of motor vehicle – {t._inv()}",
                lambda: f"Loss on disposal of plant and machinery – {t._month()} {t._year()}",
                lambda: f"PPE disposal gain – office equipment – {t._year()}",
                lambda: f"Fixed asset disposal proceeds – net – {t._year()}",
                lambda: f"Sale of IT equipment – gain/(loss) – {t._inv()}",
                lambda: f"Property disposal proceeds – net book value adjustment",
                lambda: f"Vehicle fleet disposal – gain on sale – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL023 – Gain / (loss) on negative goodwill / bargain purchases
            # ----------------------------------------------------------------
            "DPL023": [
                lambda: f"Negative goodwill released – {t._sub()} acquisition – {t._year()}",
                lambda: f"Bargain purchase gain on {t._sub()} acquisition – {t._year()}",
                lambda: f"Excess of fair value over purchase consideration – {t._year()}",
                lambda: f"IFRS 3 negative goodwill recognised – {t._year()}",
                lambda: f"Bargain purchase element – {t._sub()} – {t._year()}",
                lambda: f"Negative goodwill credit – business combination – {t._year()}",
                lambda: f"IFRS 3 day-one bargain purchase gain – {t._sub()} – {t._year()}",
                lambda: f"Net assets exceed acquisition price – negative goodwill – {t._year()}",
                lambda: f"Excess fair value recognised in P&L – {t._sub()} – {t._year()}",
                lambda: f"Business combination gain – identifiable assets exceed consideration – {t._year()}",
                lambda: f"Badwill credit – {t._sub()} acquisition – {t._year()}",
                lambda: f"Negative goodwill income – {t._year()}",
                lambda: f"Acquired net assets exceed purchase price – {t._year()}",
                lambda: f"IFRS 3 reassessment gain – {t._sub()} – {t._year()}",
                lambda: f"Fair value surplus on acquisition – released to profit – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL024 – Government grant income
            # ----------------------------------------------------------------
            "DPL024": [
                lambda: f"{t._inv()} – Government grant received – {t._grant()}",
                lambda: f"{t._grant()} grant income – {t._month()} {t._year()}",
                lambda: f"Export grant credit – {t._grant()}",
                lambda: f"R&D tax credit received – {t._year()}",
                lambda: f"Business development grant – {t._year()}",
                lambda: f"Capital grant amortisation – {t._month()} {t._year()}",
                lambda: f"Enterprise zone grant income – {t._year()}",
                lambda: f"Freeport incentive grant received – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL025 – Health and safety costs
            # ----------------------------------------------------------------
            "DPL025": [
                lambda: f"H&S training costs – {t._month()} {t._year()}",
                lambda: f"Safety equipment purchase – {t._dept()}",
                lambda: f"First aid training – staff – {t._month()}",
                lambda: f"Fire safety inspection expense – {t._loc()}",
                lambda: f"PPE purchase – health and safety compliance – {t._year()}",
                lambda: f"Risk assessment costs – {t._gen()}",
                lambda: f"Occupational health service – {t._month()} {t._year()}",
                lambda: f"Safety audit fees – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL026 – Hyper-inflation gains / (losses)
            # ----------------------------------------------------------------
            "DPL026": [
                lambda: f"Hyperinflationary adjustment – overseas subsidiary – {t._year()}",
                lambda: f"IAS 29 hyperinflation restatement gain – {t._year()}",
                lambda: f"Monetary position loss – hyperinflationary economy – {t._year()}",
                lambda: f"Hyperinflation restatement – {t._year()}",
                lambda: f"IAS 29 indexation gain – {t._month()} {t._year()}",
                lambda: f"Net monetary position – hyperinflation adjustment",
                lambda: f"IAS 29 general price level adjustment – {t._year()}",
                lambda: f"Hyperinflationary economy restatement – {t._sub()} – {t._year()}",
                lambda: f"Purchasing power gain – IAS 29 – {t._year()}",
                lambda: f"Hyperinflation monetary item loss – {t._year()}",
                lambda: f"IAS 29 restatement of non-monetary assets – {t._year()}",
                lambda: f"High inflation economy adjustment – {t._year()}",
                lambda: f"Cumulative price index restatement – {t._sub()} – {t._year()}",
                lambda: f"Hyperinflationary subsidiary restatement gain/(loss) – {t._year()}",
                lambda: f"Net purchasing power loss on monetary items – {t._year()}",
                lambda: f"IAS 29 hyperinflation adjustment – {t._ccy()} functional currency – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL027 – Impairment loss (reversal) on investments
            # ----------------------------------------------------------------
            "DPL027": [
                lambda: f"Impairment of investment in {t._sub()} – {t._year()}",
                lambda: f"Investment write-down – {t._sub()} shares – {t._year()}",
                lambda: f"Impairment reversal – subsidiary investment recovery – {t._year()}",
                lambda: f"Investment impairment charge – {t._year()}",
                lambda: f"Available-for-sale investment impairment – {t._year()}",
                lambda: f"Equity investment write-down – {t._sub()} – {t._month()}",
                lambda: f"Investment carrying value reduction – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL028 – Impairment loss / (reversal), intangible assets
            # ----------------------------------------------------------------
            "DPL028": [
                lambda: f"Goodwill impairment charge – {t._sub()} CGU – {t._year()}",
                lambda: f"Software impairment loss – {t._it()} – {t._year()}",
                lambda: f"Customer relationship intangible impairment – {t._year()}",
                lambda: f"Trademark impairment charge – {t._year()}",
                lambda: f"Intangible asset impairment reversal – {t._year()}",
                lambda: f"Brand name impairment – {t._sub()} CGU review – {t._year()}",
                lambda: f"IAS 36 goodwill impairment – {t._sub()} – {t._year()}",
                lambda: f"Customer list impairment charge – {t._year()}",
                lambda: f"Patent impairment loss – {t._year()}",
                lambda: f"Technology intangible impairment – {t._year()}",
                lambda: f"Acquired intangible impairment – {t._sub()} – {t._year()}",
                lambda: f"Impairment reversal – brand name – {t._year()}",
                lambda: f"CGU impairment – {t._sub()} intangibles – {t._year()}",
                lambda: f"ERP system impairment write-down – {t._it()} – {t._year()}",
                lambda: f"Impairment – capitalised software – {t._year()}",
                lambda: f"Goodwill write-down – {t._sub()} CGU – {t._year()}",
                lambda: f"Customer relationships impairment reversal – {t._year()}",
                lambda: f"Domain name impairment – {t._year()}",
                lambda: f"Franchise intangible impairment charge – {t._year()}",
                lambda: f"IAS 36 annual impairment test charge – {t._sub()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL029 – Impairment loss / (reversal), property, plant and equipment
            # ----------------------------------------------------------------
            "DPL029": [
                lambda: f"Property impairment charge – {t._loc()} – {t._year()}",
                lambda: f"PPE impairment reversal – {t._year()}",
                lambda: f"Plant and machinery impairment charge – {t._year()}",
                lambda: f"Leasehold improvements impairment – {t._loc()} – {t._year()}",
                lambda: f"Right-of-use asset impairment – {t._year()}",
                lambda: f"Asset impairment charge – {t._dept()} equipment – {t._year()}",
                lambda: f"IAS 36 PPE impairment – {t._loc()} – {t._year()}",
                lambda: f"Manufacturing plant impairment – {t._year()}",
                lambda: f"Office fit-out impairment charge – {t._loc()} – {t._year()}",
                lambda: f"Vehicle fleet impairment – {t._year()}",
                lambda: f"Warehouse impairment charge – {t._loc()} – {t._year()}",
                lambda: f"IFRS 16 right-of-use asset impairment – {t._year()}",
                lambda: f"PPE impairment reversal – {t._loc()} – {t._year()}",
                lambda: f"Tangible fixed asset write-down – {t._year()}",
                lambda: f"IAS 36 impairment review – property – {t._year()}",
                lambda: f"Construction in progress impairment – {t._year()}",
                lambda: f"Lab equipment impairment charge – {t._year()}",
                lambda: f"Server hardware impairment – {t._it()} – {t._year()}",
                lambda: f"CGU tangible asset impairment – {t._sub()} – {t._year()}",
                lambda: f"Asset impairment reversal – PPE – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL030 – Income from other investments
            # ----------------------------------------------------------------
            "DPL030": [
                lambda: f"Dividend income from {t._sub()} – {t._month()} {t._year()}",
                lambda: f"Income from equity investments – {t._year()}",
                lambda: f"Return on venture capital investment – {t._year()}",
                lambda: f"Investment income – minority shareholding – {t._month()}",
                lambda: f"Profit share from joint venture – {t._year()}",
                lambda: f"Listed equity dividend received – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL031 – Income tax expense / (credit)
            # ----------------------------------------------------------------
            "DPL031": [
                lambda: f"Corporation tax charge for year ended {t._year()}",
                lambda: f"Deferred tax liability movement – {t._year()}",
                lambda: f"Income tax expense – current year – {t._year()}",
                lambda: f"Tax provision adjustment – {t._month()} {t._year()}",
                lambda: f"Deferred tax credit on temporary differences – {t._year()}",
                lambda: f"Current tax charge – {t._year()} return",
                lambda: f"Overseas withholding tax – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL032 – Increase / (decrease) in stocks / inventories
            # ----------------------------------------------------------------
            "DPL032": [
                lambda: f"Change in finished goods inventory – {t._month()} {t._year()}",
                lambda: f"WIP inventory movement charge – {t._year()}",
                lambda: f"Raw materials stock change – {t._month()}",
                lambda: f"Inventory increase/(decrease) adjustment – {t._year()}",
                lambda: f"Stock build – seasonal demand – {t._month()}",
                lambda: f"Inventory write-down – {t._year()}",
                lambda: f"Work in progress movement – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL033 – Insurance costs
            # ----------------------------------------------------------------
            "DPL033": [
                lambda: f"Annual insurance premium – professional indemnity – {t._year()}",
                lambda: f"{t._inv()} – {t._ins()} – property insurance",
                lambda: f"Directors and officers insurance premium – {t._year()}",
                lambda: f"Public liability insurance renewal – {t._year()}",
                lambda: f"Business insurance – {t._ins()} – {t._month()} {t._year()}",
                lambda: f"Cyber liability insurance premium – {t._year()}",
                lambda: f"Employers liability insurance – {t._year()}",
                lambda: f"Product liability insurance – {t._ins()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL034 – Inter-company recharges
            # ----------------------------------------------------------------
            "DPL034": [
                lambda: f"Management fee recharge from {t._parent()} – {t._month()} {t._year()}",
                lambda: f"Inter-company IT service recharge – {t._month()}",
                lambda: f"Group overhead recharge – {t._parent()} – {t._month()}",
                lambda: f"Shared services cost recharge – {t._dept()} – {t._year()}",
                lambda: f"Head office recharge – {t._parent()} – {t._year()}",
                lambda: f"Inter-company management charge – {t._inv()}",
                lambda: f"Group finance function recharge – {t._month()} {t._year()}",
                lambda: f"IT support recharge from parent – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL035 – Interest expense on bank overdrafts, bank loans
            # ----------------------------------------------------------------
            "DPL035": [
                lambda: f"Interest charged on {t._bank()} overdraft – {t._month()}",
                lambda: f"Loan interest expense – term facility – {t._year()}",
                lambda: f"Bank interest debit – revolving credit facility – {t._month()}",
                lambda: f"Interest on {t._bank()} term loan – {t._year()}",
                lambda: f"Overdraft interest charge – {t._bank()} – {t._month()} {t._year()}",
                lambda: f"Debt interest expense – {t._amt()} facility – {t._year()}",
                lambda: f"Senior secured loan interest – {t._month()}",
                lambda: f"Syndicated facility interest – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL036 – Irrecoverable VAT
            # ----------------------------------------------------------------
            "DPL036": [
                lambda: f"Input VAT – non-deductible – {t._month()} {t._year()}",
                lambda: f"Irrecoverable VAT on entertainment – {t._month()}",
                lambda: f"Partial exemption VAT restriction – {t._year()}",
                lambda: f"Non-recoverable VAT charge – {t._month()} {t._year()}",
                lambda: f"VAT blocked – business entertainment – {t._month()}",
                lambda: f"Blocked input tax – exempt supplies – {t._year()}",
                lambda: f"Non-business VAT apportionment – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL037 – IT and computing costs
            # ----------------------------------------------------------------
            "DPL037": [
                lambda: f"{t._it()} cloud hosting – {t._month()} subscription",
                lambda: f"Software licence renewal – {t._dept()} – {t._year()}",
                lambda: f"Annual ERP maintenance fee – {t._it()}",
                lambda: f"{t._inv()} – IT support contract – {t._it()}",
                lambda: f"SaaS subscription – {t._it()} – {t._month()} {t._year()}",
                lambda: f"Hardware maintenance – {t._dept()} servers",
                lambda: f"Cybersecurity software – annual licence – {t._year()}",
                lambda: f"IT infrastructure costs – {t._month()} {t._year()}",
                lambda: f"Data storage costs – {t._it()} – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL038 – Legal and professional costs
            # ----------------------------------------------------------------
            "DPL038": [
                lambda: f"{t._inv()} – Legal advisory fees – {t._law()}",
                lambda: f"Litigation expense – contract dispute – {t._year()}",
                lambda: f"Corporate legal services – acquisition support – {t._law()}",
                lambda: f"Employment tribunal legal costs – {t._year()}",
                lambda: f"Contract drafting fees – {t._law()} – {t._month()}",
                lambda: f"Professional fees – regulatory matter – {t._year()}",
                lambda: f"Conveyancing fees – property transaction – {t._year()}",
                lambda: f"IP legal costs – {t._law()} – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL039 – Loans written off or down
            # ----------------------------------------------------------------
            "DPL039": [
                lambda: f"Loan written off – {t._sub()} – {t._year()}",
                lambda: f"Director loan impairment charge – {t._year()}",
                lambda: f"Related party loan write-down – {t._year()}",
                lambda: f"Intercompany loan written off – {t._sub()} – {t._year()}",
                lambda: f"Loan receivable write-off – {t._inv()}",
                lambda: f"Credit impairment – inter-company loan – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL040 – Non-cash assets declared for distribution to owners
            # ----------------------------------------------------------------
            "DPL040": [
                lambda: f"Non-cash asset distribution declared – {t._year()}",
                lambda: f"In-specie dividend – property distribution – {t._year()}",
                lambda: f"Dividend in kind – investment portfolio – {t._year()}",
                lambda: f"Non-cash dividend – {t._sub()} shares – {t._year()}",
                lambda: f"Asset distribution to shareholders – {t._year()}",
                lambda: f"In-specie transfer of {t._asset()} to shareholders – {t._year()}",
                lambda: f"Non-monetary dividend declared – {t._year()}",
                lambda: f"Distribution in kind – {t._sub()} equity – {t._year()}",
                lambda: f"IFRIC 17 non-cash asset distribution – {t._year()}",
                lambda: f"Property dividend declared to owners – {t._year()}",
                lambda: f"Distribution of {t._asset()} to equity holders – {t._year()}",
                lambda: f"In-specie return of capital – {t._year()}",
                lambda: f"Non-cash return of value to shareholders – {t._year()}",
                lambda: f"Asset demerger – distribution to shareholders – {t._year()}",
                lambda: f"Non-monetary distribution – fair value measurement – {t._year()}",
                lambda: f"Spin-off distribution of {t._sub()} shares – {t._year()}",
                lambda: f"In-specie dividend – commercial property – {t._year()}",
                lambda: f"Non-cash asset declared for distribution – board resolution – {t._year()}",
                lambda: f"Distribution of investment property to owners – {t._year()}",
                lambda: f"IFRIC 17 fair value liability recognised – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL041 – Other entertaining costs
            # ----------------------------------------------------------------
            "DPL041": [
                lambda: f"Client entertainment – {t._month()} {t._year()}",
                lambda: f"Staff Christmas party expense – {t._year()}",
                lambda: f"Business entertainment – restaurant – {t._inv()}",
                lambda: f"Corporate hospitality costs – {t._month()}",
                lambda: f"Client dinner – {t._client()} – {t._month()}",
                lambda: f"Team away day expense – {t._dept()} – {t._year()}",
                lambda: f"Sporting event hospitality – {t._month()} {t._year()}",
                lambda: f"Staff entertainment – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL042 – Other expenses transferred to balance sheet
            # ----------------------------------------------------------------
            "DPL042": [
                lambda: f"Expense capitalised to work in progress – {t._month()} {t._year()}",
                lambda: f"Cost transferred to fixed asset register – {t._year()}",
                lambda: f"Development cost capitalisation – {t._month()}",
                lambda: f"Pre-production cost transferred to balance sheet – {t._year()}",
                lambda: f"Revenue cost reclassified to capital – {t._inv()}",
                lambda: f"Expenditure transferred to {t._proj()} project balance sheet",
            ],

            # ----------------------------------------------------------------
            # DPL043 – Other foreign exchange loss / (gain)
            # ----------------------------------------------------------------
            "DPL043": [
                lambda: f"FX loss on USD settlement – AP {t._inv()}",
                lambda: f"Foreign exchange revaluation gain – {t._month()} {t._year()}",
                lambda: f"Currency translation loss on EUR balances – {t._month()}",
                lambda: f"FX gain on EUR receivable settlement – {t._year()}",
                lambda: f"Unrealised FX gain – balance sheet revaluation – {t._month()}",
                lambda: f"Realised foreign exchange loss – {t._inv()}",
                lambda: f"Currency translation difference – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL044 – Other non-operating net gain (loss), before tax
            # ----------------------------------------------------------------
            "DPL044": [
                lambda: f"Non-operating gain – settlement received – {t._year()}",
                lambda: f"Non-recurring income – {t._year()}",
                lambda: f"Insurance claim receipt – property damage – {t._year()}",
                lambda: f"Non-operating loss – {t._year()}",
                lambda: f"One-off non-operating gain – {t._month()} {t._year()}",
                lambda: f"Exceptional non-operating item – {t._year()}",
                lambda: f"Non-operating net gain before tax – {t._year()}",
                lambda: f"Windfall receipt – non-recurring – {t._year()}",
                lambda: f"Exceptional item – non-operating – {t._year()}",
                lambda: f"Pre-tax non-operating loss – litigation settlement – {t._year()}",
                lambda: f"Non-operating gain on compulsory purchase – {t._year()}",
                lambda: f"Extraordinary income – non-trading – {t._year()}",
                lambda: f"Compensation received – non-operating – {t._year()}",
                lambda: f"Non-operating item – insurance recovery – {t._year()}",
                lambda: f"Non-trading gain before tax – {t._month()} {t._year()}",
                lambda: f"Windfall non-operating settlement – {t._year()}",
                lambda: f"Pre-tax gain on non-operating activity – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL045 – Other operating income
            # ----------------------------------------------------------------
            "DPL045": [
                lambda: f"Other operating income – service recharge – {t._month()}",
                lambda: f"Miscellaneous operating income received – {t._year()}",
                lambda: f"Operational income adjustment – {t._month()} {t._year()}",
                lambda: f"Income from ancillary activities – {t._year()}",
                lambda: f"Sundry operating income – {t._month()}",
                lambda: f"Operating income – reimbursement received – {t._inv()}",
                lambda: f"Service fee income – {t._client()} – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL046 – Other operational and administration costs
            # Key differentiator: operational AND administrative overhead
            # ----------------------------------------------------------------
            "DPL046": [
                lambda: f"Office supplies and administration costs – {t._month()} {t._year()}",
                lambda: f"General administration overhead – {t._dept()} – {t._year()}",
                lambda: f"Operational and admin overhead allocation – {t._dept()}",
                lambda: f"Operational administration costs – {t._month()} {t._year()}",
                lambda: f"General admin overhead charge – {t._month()}",
                lambda: f"Administration and operational costs – {t._dept()} – {t._inv()}",
                lambda: f"Admin services overhead – {t._gen()} – {t._year()}",
                lambda: f"Operational overhead – administration and facilities – {t._year()}",
                lambda: f"Back-office administration costs – {t._month()} {t._year()}",
                lambda: f"Premises and operational admin expense – {t._year()}",
                lambda: f"Corporate administration overhead – {t._dept()} – {t._year()}",
                lambda: f"Running costs – operational administration – {t._month()}",
                lambda: f"Day-to-day operational admin expenses – {t._dept()}",
                lambda: f"Miscellaneous operational and admin charge – {t._year()}",
                lambda: f"Admin support costs – {t._dept()} – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL047 – Other repairs and maintenance expense
            # ----------------------------------------------------------------
            "DPL047": [
                lambda: f"Building repairs and maintenance – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Facility repair costs – {t._gen()}",
                lambda: f"Plant and equipment maintenance – {t._month()}",
                lambda: f"Vehicle repair costs – {t._month()}",
                lambda: f"Property maintenance charge – {t._loc()}",
                lambda: f"HVAC system maintenance – {t._loc()} – {t._year()}",
                lambda: f"Lift maintenance contract – {t._year()}",
                lambda: f"General repairs – {t._loc()} – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL048 – Other staff costs
            # ----------------------------------------------------------------
            "DPL048": [
                lambda: f"Employee benefits expense – {t._month()} {t._year()}",
                lambda: f"Staff welfare costs – {t._dept()}",
                lambda: f"Training and development – {t._dept()} – {t._year()}",
                lambda: f"Recruitment fees – {t._dept()} – {t._year()}",
                lambda: f"Employee incentive scheme cost – {t._year()}",
                lambda: f"Staff medical benefit – {t._month()}",
                lambda: f"Employee assistance programme – {t._year()}",
                lambda: f"Relocation expenses – staff – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL049 – Outsourced services costs
            # ----------------------------------------------------------------
            "DPL049": [
                lambda: f"Outsourced payroll processing fees – {t._month()} {t._year()}",
                lambda: f"{t._gen()} – managed services contract – {t._month()}",
                lambda: f"Outsourced IT support – {t._month()} {t._year()}",
                lambda: f"Business process outsourcing charge – {t._year()}",
                lambda: f"Facilities management outsourcing – {t._gen()}",
                lambda: f"Outsourced HR services – {t._month()} {t._year()}",
                lambda: f"Customer contact centre outsourcing – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL050 – Own work capitalised
            # ----------------------------------------------------------------
            "DPL050": [
                lambda: f"Internal labour capitalised – {t._proj()} project – {t._month()}",
                lambda: f"Own work capitalised – software development – {t._year()}",
                lambda: f"Staff costs transferred to capital project – {t._month()} {t._year()}",
                lambda: f"Internal resources capitalised – {t._proj()} – {t._month()} {t._year()}",
                lambda: f"Own labour – asset construction – {t._year()}",
                lambda: f"Capitalised internal engineering costs – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL051 – Pension costs, defined benefit plan
            # ----------------------------------------------------------------
            "DPL051": [
                lambda: f"Defined benefit pension scheme – service cost – {t._month()} {t._year()}",
                lambda: f"Actuarial valuation costs – pension scheme – {t._year()}",
                lambda: f"DB pension current service charge – {t._year()}",
                lambda: f"Pension scheme contribution – defined benefit – {t._month()}",
                lambda: f"Past service cost – defined benefit plan – {t._year()}",
                lambda: f"DB scheme deficit contribution – {t._month()} {t._year()}",
                lambda: f"Defined benefit pension – admin costs – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL052 – Pension costs, defined contribution plan
            # ----------------------------------------------------------------
            "DPL052": [
                lambda: f"Auto-enrolment pension contribution – {t._month()} {t._year()}",
                lambda: f"Employer pension contribution – defined contribution – {t._month()}",
                lambda: f"Group pension scheme payments – {t._month()} {t._year()}",
                lambda: f"Workplace pension – {t._month()} {t._year()}",
                lambda: f"DC pension employer contribution – {t._dept()} – {t._month()}",
                lambda: f"NEST / defined contribution pension – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL053 – Political donations
            # ----------------------------------------------------------------
            "DPL053": [
                lambda: f"Political party donation – {t._party()} – {t._year()}",
                lambda: f"Donation to {t._party()} – disclosed in accounts – {t._year()}",
                lambda: f"Political donation – {t._party()} – {t._year()}",
                lambda: f"Campaign contribution – {t._party()} – {t._year()}",
                lambda: f"Political party contribution – {t._year()}",
                lambda: f"Party political donation – {t._party()} – {t._year()}",
                lambda: f"Political campaign donation – {t._year()}",
                lambda: f"CA 2006 s.366 political donation – {t._party()} – {t._year()}",
                lambda: f"Electoral campaign contribution – {t._year()}",
                lambda: f"Political expenditure – {t._party()} – {t._year()}",
                lambda: f"Donation to political organisation – {t._year()}",
                lambda: f"Statutory political donation disclosure – {t._year()}",
                lambda: f"Political party subscription – {t._party()} – {t._year()}",
                lambda: f"Election campaign donation – {t._year()}",
                lambda: f"Policy foundation donation – {t._year()}",
                lambda: f"Political think tank contribution – {t._year()}",
                lambda: f"Party membership and political donation – {t._year()}",
                lambda: f"Political lobbying contribution – {t._year()}",
                lambda: f"Donation disclosed under CA 2006 – {t._party()} – {t._year()}",
                lambda: f"Political party funding – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL054 – Printing, postage and stationery costs
            # ----------------------------------------------------------------
            "DPL054": [
                lambda: f"Printing and stationery – {t._dept()} – {t._month()} {t._year()}",
                lambda: f"Postage and courier costs – {t._month()} {t._year()}",
                lambda: f"Office supplies printing – {t._inv()}",
                lambda: f"Stationery purchase – {t._gen()} – {t._year()}",
                lambda: f"Document printing and binding costs – {t._month()}",
                lambda: f"Franking machine and postage – {t._month()}",
                lambda: f"Print room costs – {t._loc()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL055 – Rent, rates and services costs
            # ----------------------------------------------------------------
            "DPL055": [
                lambda: f"Office rent – {t._loc()} – {t._month()} {t._year()}",
                lambda: f"Warehouse lease payment – {t._month()} {t._year()}",
                lambda: f"Property rental expense – {t._loc()}",
                lambda: f"Business rates – {t._loc()} – {t._quarter()} {t._year()}",
                lambda: f"Service charge – {t._loc()} – {t._month()}",
                lambda: f"Leasehold rent – {t._loc()} – {t._year()}",
                lambda: f"IFRS 16 lease liability – interest and depreciation – {t._month()}",
                lambda: f"Rent review uplift – {t._loc()} – {t._year()}",
                lambda: f"Ground rent payment – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL056 – Research and development expense
            # ----------------------------------------------------------------
            "DPL056": [
                lambda: f"R&D project costs – {t._proj()} – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – R&D materials and testing",
                lambda: f"Research expenditure – innovation lab – {t._year()}",
                lambda: f"Development costs – {t._proj()} product – {t._year()}",
                lambda: f"Clinical trial expense – {t._year()}",
                lambda: f"R&D staff costs – {t._dept()} – {t._month()}",
                lambda: f"Prototype development expense – {t._proj()} – {t._year()}",
                lambda: f"R&D subcontractor costs – {t._gen()}",
            ],

            # ----------------------------------------------------------------
            # DPL057 – Residual finance COSTS (payable / expense)
            # Key differentiator: COST / CHARGE / EXPENSE / PAYABLE
            # ----------------------------------------------------------------
            "DPL057": [
                lambda: f"Finance CHARGE – lease liability – {t._month()} {t._year()}",
                lambda: f"Commitment FEE on undrawn facility – {t._year()}",
                lambda: f"Residual finance COSTS – {t._month()} {t._year()}",
                lambda: f"Amortisation of debt issuance COSTS – {t._year()}",
                lambda: f"Financing COSTS – subordinated debt – {t._year()}",
                lambda: f"Finance COST – put/call option liability – {t._year()}",
                lambda: f"Deferred consideration interest EXPENSE – {t._year()}",
                lambda: f"Non-bank finance CHARGE – {t._month()} {t._year()}",
                lambda: f"Preference share dividend treated as finance COST – {t._year()}",
                lambda: f"Finance COST – vendor loan note – {t._year()}",
                lambda: f"Loan arrangement FEE amortisation – {t._year()}",
                lambda: f"Finance EXPENSE – lease – IFRS 16 – {t._month()} {t._year()}",
                lambda: f"Other residual finance COSTS – {t._year()}",
                lambda: f"Accretion of discount on provision – finance COST – {t._year()}",
                lambda: f"Mezzanine debt finance CHARGE – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL058 – Residual finance INCOME (receivable)
            # Key differentiator: INCOME / RECEIVED / RECEIVABLE / CREDIT
            # ----------------------------------------------------------------
            "DPL058": [
                lambda: f"Finance lease RECEIVABLE interest INCOME – {t._month()} {t._year()}",
                lambda: f"Other finance INCOME – pension surplus – {t._year()}",
                lambda: f"Unwinding of fair value adjustment – finance INCOME – {t._year()}",
                lambda: f"Finance INCOME – employee benefit plan – {t._year()}",
                lambda: f"Expected return on plan assets – DB pension INCOME – {t._year()}",
                lambda: f"Deferred consideration – unwinding of discount – INCOME – {t._year()}",
                lambda: f"Residual finance INCOME – {t._month()} {t._year()}",
                lambda: f"Finance INCOME – vendor loan receivable – {t._year()}",
                lambda: f"Net interest INCOME – defined benefit pension asset – {t._year()}",
                lambda: f"Finance INCOME on discounted receivable – {t._year()}",
                lambda: f"Other finance INCOME – {t._month()} {t._year()}",
                lambda: f"Accretion INCOME – finance receivable – {t._year()}",
                lambda: f"Finance INCOME – put option asset unwind – {t._year()}",
                lambda: f"Net finance INCOME – pension scheme surplus – {t._year()}",
                lambda: f"Residual finance INCOME credited – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL059 – Restructuring costs
            # ----------------------------------------------------------------
            "DPL059": [
                lambda: f"Restructuring provision charge – {t._year()}",
                lambda: f"Redundancy costs – {t._dept()} – {t._year()}",
                lambda: f"Office closure costs – {t._loc()} – {t._year()}",
                lambda: f"Employee redundancy payments – {t._month()} {t._year()}",
                lambda: f"Business reorganisation expense – {t._year()}",
                lambda: f"Restructuring advisory fees – {t._firm()} – {t._year()}",
                lambda: f"Site rationalisation costs – {t._loc()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL060 – Royalties or similar payments for intellectual property
            # ----------------------------------------------------------------
            "DPL060": [
                lambda: f"Royalty payment to {t._iph()} – {t._month()} {t._year()}",
                lambda: f"Licence fee for use of {t._ip()} – {t._year()}",
                lambda: f"Patent licence royalties – {t._month()}",
                lambda: f"Music royalty costs – {t._month()} {t._year()}",
                lambda: f"Trade mark licence payment – {t._iph()} – {t._year()}",
                lambda: f"IP royalty – {t._inv()} – {t._year()}",
                lambda: f"Franchise fee payment – {t._year()}",
                lambda: f"Technology licence fee – {t._iph()} – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL061 – Security costs
            # ----------------------------------------------------------------
            "DPL061": [
                lambda: f"Building security services – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Security contractor – {t._gen()}",
                lambda: f"CCTV maintenance and monitoring – {t._loc()}",
                lambda: f"Manned guarding services – {t._month()} {t._year()}",
                lambda: f"Cyber security licence and services – {t._year()}",
                lambda: f"Access control system maintenance – {t._year()}",
                lambda: f"Security patrol costs – {t._loc()} – {t._month()}",
            ],

            # ----------------------------------------------------------------
            # DPL062 – Share-based payment expense, equity settled
            # ----------------------------------------------------------------
            "DPL062": [
                lambda: f"Share option expense – IFRS 2 charge – {t._month()} {t._year()}",
                lambda: f"Employee share plan charge – {t._year()}",
                lambda: f"Long-term incentive plan (LTIP) charge – {t._year()}",
                lambda: f"RSU vesting expense – {t._month()} {t._year()}",
                lambda: f"Equity-settled SBP charge – {t._year()}",
                lambda: f"SAYE scheme cost – {t._year()}",
                lambda: f"Share award charge – {t._dept()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL063 – Subscriptions costs
            # ----------------------------------------------------------------
            "DPL063": [
                lambda: f"Annual subscription – {t._subv()} – {t._year()}",
                lambda: f"Professional body membership – {t._month()} {t._year()}",
                lambda: f"Trade association subscription – {t._year()}",
                lambda: f"Software subscription – {t._it()} – {t._month()} {t._year()}",
                lambda: f"Journal and publication subscription – {t._year()}",
                lambda: f"Online data subscription – {t._subv()} – {t._year()}",
                lambda: f"Professional membership renewal – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL064 – Telecommunications costs
            # ----------------------------------------------------------------
            "DPL064": [
                lambda: f"Mobile phone contract costs – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Broadband and telephony – {t._tel()}",
                lambda: f"Data centre connectivity charges – {t._year()}",
                lambda: f"VoIP telephony costs – {t._month()} {t._year()}",
                lambda: f"Corporate mobile plan – {t._dept()} – {t._month()}",
                lambda: f"Fixed line rental – {t._tel()} – {t._year()}",
                lambda: f"International call charges – {t._month()}",
                lambda: f"Satellite communications – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL065 – Travel and subsistence costs
            # ----------------------------------------------------------------
            "DPL065": [
                lambda: f"Flight booking – business trip – {t._month()} {t._year()}",
                lambda: f"Hotel accommodation – client visit – {t._month()}",
                lambda: f"Travel reimbursement – employee expense claim – {t._inv()}",
                lambda: f"Rail travel – {t._dept()} staff – {t._month()}",
                lambda: f"Taxi and car hire – {t._month()} {t._year()}",
                lambda: f"Subsistence expenses – {t._dept()} – {t._month()}",
                lambda: f"Mileage claim – {t._inv()} – {t._month()}",
                lambda: f"Overseas travel – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL066 – Turnover / revenue
            # ----------------------------------------------------------------
            "DPL066": [
                lambda: f"Sales revenue – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Service income – {t._client()}",
                lambda: f"Product sales – {t._dept()} – {t._month()} {t._year()}",
                lambda: f"Professional services income – {t._month()} {t._year()}",
                lambda: f"Revenue from contracts with customers – {t._year()}",
                lambda: f"Consulting fee income – {t._client()} – {t._inv()}",
                lambda: f"Subscription revenue – {t._month()} {t._year()}",
                lambda: f"Licence fee income – {t._client()} – {t._year()}",
                lambda: f"Project completion revenue – {t._proj()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL067 – Unwinding of discount on provisions, expense
            # Key differentiator: unwinding / discount / provision / time-value
            # ----------------------------------------------------------------
            "DPL067": [
                lambda: f"Unwinding of discount – decommissioning provision – {t._year()}",
                lambda: f"Discount unwind – long-term provision – {t._year()}",
                lambda: f"Present value unwinding – {t._prov()} – {t._month()} {t._year()}",
                lambda: f"Finance cost – unwinding of provision discount – {t._year()}",
                lambda: f"IAS 37 provision discount unwind – {t._year()}",
                lambda: f"Time value of money charge – {t._prov()} – {t._year()}",
                lambda: f"Provision discount unwind – {t._prov()} – {t._year()}",
                lambda: f"Unwinding of present value adjustment – {t._prov()} – {t._year()}",
                lambda: f"Accretion of discount – {t._prov()} – {t._year()}",
                lambda: f"IAS 37 time value of money – provision – {t._year()}",
                lambda: f"Discount unwind – {t._prov()} – {t._month()} {t._year()}",
                lambda: f"Finance expense – unwinding of discounted provision – {t._year()}",
                lambda: f"PV unwinding – {t._prov()} – {t._year()}",
                lambda: f"Discount unwind on {t._prov()} – {t._year()}",
                lambda: f"Time value unwinding – long-term liability – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL068 – Wages and salaries
            # ----------------------------------------------------------------
            "DPL068": [
                lambda: f"Monthly payroll – {t._month()} {t._year()}",
                lambda: f"Salaries and wages expense – {t._dept()}",
                lambda: f"Employee payroll processing – {t._month()}",
                lambda: f"Gross wages – {t._dept()} – {t._month()} {t._year()}",
                lambda: f"Payroll run – {t._month()} {t._year()} – {t._dept()}",
                lambda: f"Staff salaries – {t._month()} {t._year()}",
                lambda: f"Bonus payroll – {t._dept()} – {t._year()}",
                lambda: f"PAYE payroll cost – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL069 – Warehouse and storage costs
            # ----------------------------------------------------------------
            "DPL069": [
                lambda: f"Warehouse storage charges – {t._month()} {t._year()}",
                lambda: f"{t._inv()} – Third party storage – {t._log()}",
                lambda: f"Cold storage facility costs – {t._month()}",
                lambda: f"Distribution centre costs – {t._month()} {t._year()}",
                lambda: f"Pallet storage fees – {t._log()} – {t._year()}",
                lambda: f"Bonded warehouse charges – {t._month()}",
                lambda: f"Self-storage unit rental – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL070 – Revenue from off payroll working
            # ----------------------------------------------------------------
            "DPL070": [
                lambda: f"Off payroll worker fees – IR35 – {t._month()} {t._year()}",
                lambda: f"Contractor income – off payroll engagement – {t._year()}",
                lambda: f"IR35 deemed employment receipt – {t._month()}",
                lambda: f"Off payroll revenue – {t._inv()} – {t._year()}",
                lambda: f"Deemed employment income – IR35 – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL071 – Coronavirus job retention scheme income
            # ----------------------------------------------------------------
            "DPL071": [
                lambda: f"CJRS furlough grant received – {t._month()} {t._year()}",
                lambda: f"Coronavirus job retention scheme income – {t._year()}",
                lambda: f"Furlough claim receipt – HMRC – {t._month()}",
                lambda: f"COVID-19 wage support grant – {t._year()}",
                lambda: f"CJRS claim – {t._dept()} – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL072 – Other coronavirus grants
            # ----------------------------------------------------------------
            "DPL072": [
                lambda: f"Coronavirus business interruption grant – {t._year()}",
                lambda: f"COVID-19 local authority grant – {t._year()}",
                lambda: f"Restart grant received – {t._month()} {t._year()}",
                lambda: f"Business rates relief – coronavirus – {t._year()}",
                lambda: f"CBILS loan / BBLS government support received – {t._year()}",
                lambda: f"Other COVID-19 grant income – {t._year()}",
                lambda: f"SEISS self-employment income support – {t._year()}",
                lambda: f"COVID-19 Additional Restrictions grant – {t._year()}",
                lambda: f"Coronavirus discretionary grant – local council – {t._year()}",
                lambda: f"COVID-19 Bounce Back Loan Scheme income – {t._year()}",
                lambda: f"Hospitality sector grant – coronavirus – {t._year()}",
                lambda: f"Local Restrictions Support grant – COVID-19 – {t._year()}",
                lambda: f"COVID-19 business support grant – {t._year()}",
                lambda: f"Retail, hospitality and leisure grant – COVID-19 – {t._year()}",
                lambda: f"COVID-19 resilience grant – {t._year()}",
                lambda: f"Coronavirus emergency support grant – {t._year()}",
                lambda: f"Test and trace support payment – COVID-19 – {t._year()}",
                lambda: f"Other pandemic support grant received – {t._year()}",
                lambda: f"COVID-19 recovery grant – {t._year()}",
                lambda: f"Pandemic business interruption grant – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL073 – Off payroll working expense
            # ----------------------------------------------------------------
            "DPL073": [
                lambda: f"Off payroll contractor cost – {t._inv()}",
                lambda: f"IR35 worker engagement cost – {t._month()} {t._year()}",
                lambda: f"Deemed employed contractor payment – {t._year()}",
                lambda: f"Off payroll working costs – {t._dept()} – {t._month()}",
                lambda: f"IR35 deemed employee cost – {t._inv()} – {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL074 – Other costs
            # Key differentiator: unclassified / sundry / catch-all costs
            # ----------------------------------------------------------------
            "DPL074": [
                lambda: f"Miscellaneous unallocated expense – {t._month()} {t._year()}",
                lambda: f"Sundry costs – catch-all – {t._dept()} – {t._year()}",
                lambda: f"Other unclassified cost – {t._month()}",
                lambda: f"Sundry charges – {t._year()}",
                lambda: f"Unallocated expense – {t._inv()}",
                lambda: f"Miscellaneous other cost – {t._gen()} – {t._month()}",
                lambda: f"Unclassified expenditure – {t._dept()} – {t._year()}",
                lambda: f"Catch-all cost allocation – {t._month()} {t._year()}",
                lambda: f"Other costs – not elsewhere classified – {t._year()}",
                lambda: f"Residual cost – unclassified – {t._month()}",
                lambda: f"General sundry expense – {t._year()}",
                lambda: f"Miscellaneous cost – no specific category – {t._year()}",
                lambda: f"Other charges – unspecified – {t._inv()}",
                lambda: f"Sundry other expense – {t._dept()} – {t._year()}",
                lambda: f"Unallocated other costs – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL075 – Non-bank interest and similar INCOME receivable
            # Key differentiator: INCOME / RECEIVED / RECEIVABLE (non-bank source)
            # ----------------------------------------------------------------
            "DPL075": [
                lambda: f"Interest RECEIVED on shareholder loan – {t._month()} {t._year()}",
                lambda: f"Non-bank interest INCOME – related party – {t._year()}",
                lambda: f"Loan note interest RECEIVED – {t._month()}",
                lambda: f"Interest INCOME on inter-company balance – {t._month()} {t._year()}",
                lambda: f"Inter-company loan interest INCOME – {t._sub()} – {t._year()}",
                lambda: f"Related party interest RECEIVABLE – {t._year()}",
                lambda: f"Non-bank interest RECEIVABLE – {t._sub()} – {t._year()}",
                lambda: f"Shareholder loan interest INCOME RECEIVED – {t._year()}",
                lambda: f"Intra-group interest INCOME – {t._sub()} – {t._month()}",
                lambda: f"Loan note interest INCOME – {t._year()}",
                lambda: f"Non-bank interest and similar INCOME – {t._month()} {t._year()}",
                lambda: f"Inter-company interest RECEIVABLE – {t._sub()} – {t._year()}",
                lambda: f"Related party loan interest CREDITED – {t._year()}",
                lambda: f"Subordinated loan interest RECEIVED – {t._year()}",
                lambda: f"Non-bank interest INCOME receivable – {t._month()} {t._year()}",
            ],

            # ----------------------------------------------------------------
            # DPL076 – Non-bank interest and similar CHARGES
            # Key differentiator: CHARGE / PAYABLE / EXPENSE / COST (non-bank source)
            # ----------------------------------------------------------------
            "DPL076": [
                lambda: f"Interest CHARGE on shareholder loan – {t._month()} {t._year()}",
                lambda: f"Non-bank finance CHARGES – {t._sub()} – {t._year()}",
                lambda: f"Loan note interest PAYABLE – {t._year()}",
                lambda: f"Interest PAYABLE on related party loan – {t._month()}",
                lambda: f"Inter-company interest EXPENSE – {t._sub()} – {t._year()}",
                lambda: f"Related party interest PAYABLE – {t._year()}",
                lambda: f"Non-bank interest CHARGE – {t._sub()} – {t._year()}",
                lambda: f"Shareholder loan interest EXPENSE PAID – {t._year()}",
                lambda: f"Intra-group interest EXPENSE – {t._sub()} – {t._month()}",
                lambda: f"Loan note interest EXPENSE – {t._year()}",
                lambda: f"Non-bank interest and similar CHARGES – {t._month()} {t._year()}",
                lambda: f"Inter-company interest PAYABLE – {t._sub()} – {t._year()}",
                lambda: f"Related party loan interest CHARGED – {t._year()}",
                lambda: f"Subordinated loan interest PAYABLE – {t._year()}",
                lambda: f"Non-bank interest CHARGE payable – {t._month()} {t._year()}",
            ],
        }

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def generate_for_tag(self, tag: str, n: int = 300, max_attempts_multiplier: int = 10) -> list[dict]:
        """Generate n UNIQUE descriptions for the given tag.

        Tries up to n * max_attempts_multiplier times to reach n unique samples.
        If templates are exhausted, returns however many unique samples were found.
        """
        if tag not in self.templates:
            return []

        seen: set[str] = set()
        records: list[dict] = []
        max_attempts = n * max_attempts_multiplier

        for _ in range(max_attempts):
            if len(records) >= n:
                break
            fn = random.choice(self.templates[tag])
            desc = fn()
            if desc not in seen:
                seen.add(desc)
                records.append({"description": desc, "dpl_tag": tag})

        if len(records) < n:
            print(f"  WARNING: {tag} — only {len(records)} unique samples generated (target {n}). "
                  f"Add more templates for this tag.")
        return records

    def generate_dataset(self, tags: list[str], n_per_tag: int = 300) -> pd.DataFrame:
        all_data = []
        for tag in tags:
            all_data.extend(self.generate_for_tag(tag, n_per_tag))
        df = pd.DataFrame(all_data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split so every tag appears in all three splits."""
    train_rows, val_rows, test_rows = [], [], []

    for tag, group in df.groupby("dpl_tag"):
        g = group.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_rows.append(g.iloc[:n_train])
        val_rows.append(g.iloc[n_train : n_train + n_val])
        test_rows.append(g.iloc[n_train + n_val :])

    return (
        pd.concat(train_rows).sample(frac=1, random_state=1).reset_index(drop=True),
        pd.concat(val_rows).sample(frac=1, random_state=2).reset_index(drop=True),
        pd.concat(test_rows).sample(frac=1, random_state=3).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DPL training data.")
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="Number of samples to generate per DPL tag (default: 300).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets",
        help="Output directory for CSV files (default: datasets/).",
    )
    args = parser.parse_args()

    # All active DPL tags (DPL000 is reserved/skipped)
    all_tags = [f"DPL{i:03d}" for i in range(1, 77)]

    print(f"Generating {args.n} samples × {len(all_tags)} tags "
          f"= {args.n * len(all_tags):,} total rows …")

    generator = SyntheticDPLGenerator()
    df = generator.generate_dataset(all_tags, n_per_tag=args.n)

    train_df, val_df, test_df = split_dataset(df)

    os.makedirs(args.out, exist_ok=True)

    full_path  = os.path.join(args.out, "dpl_full.csv")
    train_path = os.path.join(args.out, "dpl_train.csv")
    val_path   = os.path.join(args.out, "dpl_val.csv")
    test_path  = os.path.join(args.out, "dpl_test.csv")

    df.to_csv(full_path,  index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"\nDataset saved to: {args.out}/")
    print(f"  Full  : {full_path}   ({len(df):,} rows)")
    print(f"  Train : {train_path}  ({len(train_df):,} rows)")
    print(f"  Val   : {val_path}    ({len(val_df):,} rows)")
    print(f"  Test  : {test_path}   ({len(test_df):,} rows)")
    print(f"\nTag distribution (first 5):")
    print(df["dpl_tag"].value_counts().head())

    print("\nSample rows:")
    print(df.sample(10, random_state=99).to_string(index=False))


if __name__ == "__main__":
    main()
