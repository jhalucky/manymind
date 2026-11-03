
"""Write a labeled KYC / bait SMS seed. Templates only. No real IDs."""

from __future__ import annotations

import csv
from pathlib import Path

SEED_PATH = Path(__file__).with_name("seed.csv")

BAIT = [
    (
        "kyc_update",
        [
            "Dear customer your KYC is pending. Update now or account will be blocked: {link}",
            "KYC expired. Verify today to keep your account active {link}",
            "Urgent KYC update required within 24 hours else services stop {link}",
            "Your bank KYC not updated. Click here immediately {link}",
            "Last chance: complete video KYC or your UPI will be stopped {link}",
            "KYC hold on your account. Confirm Aadhaar now {link}",
            "Re-KYC pending. Ignore and your debit card will be deactivated {link}",
            "Bank KYC drive. Submit documents here {link} - do not miss",
            "Your KYC is incomplete. Update Aadhaar and PAN here {link}",
            "Alert: KYC mismatch found. Fix in 2 hours {link}",
        ],
    ),
    (
        "aadhaar",
        [
            "Aadhaar not linked. Link now or DBT will stop {link}",
            "UIDAI: update your Aadhaar address immediately {link}",
            "Aadhaar KYC failed. Re-submit OTP on this page {link}",
            "Your Aadhaar will be deactivated tomorrow. Click to keep {link}",
            "Aadhaar-bank linking pending. Complete here {link}",
            "e-Aadhaar download suspended. Verify identity {link}",
            "Aadhaar OTP locked. Unlock by paying verification fee {link}",
            "Update biometric Aadhaar or ration will be cancelled {link}",
            "Aadhaar phone number mismatch. Confirm here {link}",
            "Free Aadhaar PVC card. Pay shipping only {link}",
        ],
    ),
    (
        "refund",
        [
            "Income tax refund of Rs {amount} is pending. Claim now {link}",
            "ITR refund stuck. Submit bank details here {link}",
            "You have an unclaimed refund. Verify PAN to receive {link}",
            "GST refund approved. Pay processing fee Rs {amount} {link}",
            "Your subsidy refund is ready. Click to credit {link}",
            "EPFO withdrawal pending. Confirm Aadhaar {link}",
            "Insurance claim refund Rs {amount}. Update KYC {link}",
            "Failed UPI refund waiting. Enter UPI PIN on this page {link}",
            "Government relief amount not credited. Register {link}",
            "You won a tax rebate. Share OTP to receive {link}",
        ],
    ),
    (
        "disconnect",
        [
            "Electricity will be disconnected today. Pay instantly {link}",
            "Final notice: power cut in 2 hours. Pay bill {link}",
            "Gas connection suspended. Renew KYC {link}",
            "Mobile will be barred in 3 hours. Recharge via this link {link}",
            "Water supply stop notice. Clear dues {link}",
            "Your SIM KYC expired. Re-verify or SIM off {link}",
            "Broadband discontinued tonight. Pay + update KYC {link}",
            "LPG subsidy stopped. Update Aadhaar {link}",
            "Disconnection warrant generated. Avoid by paying {link}",
            "Legal notice: utility default. Settle here {link}",
        ],
    ),
    (
        "courier",
        [
            "Courier held at customs. Pay Rs {amount} to release {link}",
            "International parcel pending duty. Pay now {link}",
            "Your package is seized. Verify ID and pay fee {link}",
            "Speed post KYC required or article returned {link}",
            "Delivery agent: share OTP and pay cash extra {link}",
            "Amazon order held. Confirm Aadhaar {link}",
            "Foreign parcel: pay GST on this official page {link}",
            "Courier KYC failed. Re-submit passport {link}",
            "Undelivered COD. Pay online to reattempt {link}",
            "Customs: item illegal unless you verify {link}",
        ],
    ),
    (
        "job_fee",
        [
            "You are selected. Pay registration fee Rs {amount} {link}",
            "Interview cleared. Deposit security to join {link}",
            "Work from home job. Pay kit charges first {link}",
            "Internship confirmed. KYC fee mandatory {link}",
            "Offer letter ready after payment of Rs {amount}",
            "HR: complete KYC and pay onboarding {link}",
            "Government job form extra fee. Pay here {link}",
            "You shortlisted. WhatsApp this number and pay token",
            "Exam passed. Pay certificate fee to proceed {link}",
            "Joining blocked until KYC payment is done {link}",
        ],
    ),
    (
        "account_block",
        [
            "Unusual login. Verify now or account blocked {link}",
            "Your UPI is frozen. Unlock by KYC {link}",
            "Card blocked for safety. Confirm CVV here {link}",
            "Net banking suspended. Reset with Aadhaar OTP {link}",
            "Account will close in 1 hour. Click to keep {link}",
            "RBI order: re-validate account instantly {link}",
            "Your cheque book request needs KYC payment {link}",
            "IMPS failed due to KYC. Update here {link}",
            "Fraud detected. Share OTP to save account {link}",
            "Do not tell anyone. Confirm PIN on this page {link}",
        ],
    ),
    (
        "upi_pressure",
        [
            "Send Rs {amount} to this UPI now or case will be filed",
            "Officer speaking. Transfer Rs {amount} for KYC clearance",
            "Pay to upi:{upi} to avoid arrest related to KYC",
            "Your family will be contacted if you do not pay Rs {amount}",
            "Secret KYC fine. Pay only on this QR",
            "I am from the bank. Send money then we refund",
            "Collect agent: pay Rs {amount} on UPI {upi} today",
            "To cancel the KYC complaint pay immediately",
            "WhatsApp video KYC. First send Rs {amount} token",
            "Last warning. UPI {upi} or we block PAN",
        ],
    ),
]

OK = [
    (
        "otp",
        [
            "{bank}: {otp} is your OTP for login. Do not share it with anyone.",
            "{bank} OTP {otp} for UPI send. Valid 10 minutes. Do not share.",
            "Your {bank} net banking OTP is {otp}. Never give this to callers.",
            "{otp} is the OTP to add a beneficiary on {bank}. Do not share.",
            "OTP {otp} for your {bank} card payment. If not you, ignore.",
            "{bank}: use {otp} to complete the transaction you started in the app.",
            "Do not share OTP. {bank} will never call and ask for {otp}.",
            "Mobile login OTP for {bank} app: {otp}. Expires soon.",
            "{bank} verification code {otp}. We will never ask you to forward this.",
            "OTP {otp} to view e-statement in the {bank} app.",
        ],
    ),
    (
        "debit_alert",
        [
            "{bank}: Rs {amount} debited from your account for UPI to a merchant.",
            "Rs {amount} spent on your {bank} debit card at a store. Not you? Use the app.",
            "{bank} A/c xx{acct}: Rs {amount} withdrawn at ATM.",
            "UPI debit Rs {amount}. Avl bal Rs {balance}. {bank}",
            "{bank}: EMI of Rs {amount} auto-debited as per your mandate.",
            "Card xx{acct} used for Rs {amount}. Open {bank} app if this was not you.",
            "{bank}: NEFT debit Rs {amount} to a registered beneficiary.",
            "Fuel purchase Rs {amount} on {bank} card. SMS for info only.",
            "{bank} standing instruction: Rs {amount} sent to your loan account.",
            "Rs {amount} paid via UPI from {bank}. No link in this message.",
        ],
    ),
    (
        "credit_alert",
        [
            "{bank}: Rs {amount} credited. Available balance Rs {balance}.",
            "Salary of Rs {amount} credited to your {bank} account.",
            "UPI received Rs {amount} from a contact. {bank}",
            "Interest Rs {amount} credited to your {bank} savings account.",
            "{bank} refund Rs {amount} processed to xx{acct}.",
            "IMPS credit Rs {amount}. {bank} xx{acct}.",
            "Cashback Rs {amount} credited. Check the {bank} app.",
            "Cheque of Rs {amount} cleared in your {bank} account.",
            "FD interest Rs {amount} posted. {bank}",
            "Rs {amount} received. This is an alert only. {bank}",
        ],
    ),
    (
        "delivery",
        [
            "Your order is out for delivery today. Pay COD to the uniformed agent only.",
            "Package arriving 5-8 pm. OTP will be in the shopper app, not this SMS.",
            "Shipment delivered. Rate the order in the shopping app.",
            "Delivery attempted. Reschedule from the original shopping app.",
            "Your refill is booked for tomorrow. No payment link in this SMS.",
            "Train ticket confirmed. Show the ticket in the official IRCTC app.",
            "Cab arriving in 5 min. PIN is in the ride app. Do not share UPI PIN.",
            "Food order picked up. Track only inside the food app.",
            "Pharmacy order packed. Delivery partner will call from a masked number.",
            "Return picked. Refund follows the app policy. We will not ask for OTP.",
        ],
    ),
    (
        "recharge_bill",
        [
            "Recharge of Rs {amount} successful on your mobile.",
            "Electricity bill of Rs {amount} paid from your {bank} UPI.",
            "DTH recharge Rs {amount} done. Next due in 30 days.",
            "Fastag recharge Rs {amount} successful.",
            "Postpaid bill Rs {amount} auto-paid. Thank you.",
            "Water bill Rs {amount} received. This SMS is a receipt.",
            "Gas booking confirmed. Pay the official delivery person only.",
            "Broadband bill Rs {amount} paid. Service continues.",
            "Insurance premium Rs {amount} received for your existing policy.",
            "School fee Rs {amount} received. Receipt is in the school app.",
        ],
    ),
    (
        "balance_info",
        [
            "{bank}: available balance Rs {balance} as of today.",
            "Mini statement request noted. Open the {bank} app to view.",
            "{bank} credit card bill of Rs {amount} generated. Pay in the official app.",
            "Your {bank} loan EMI is due on the usual date. Pay in the app.",
            "{bank}: no KYC link is sent on SMS. Visit the branch or official app.",
            "Missed call balance enquiry: Rs {balance}. {bank}",
            "Your {bank} relationship manager will never ask for OTP or PIN.",
            "{bank} holiday tomorrow. Digital channels work as usual.",
            "Card xx{acct} due date reminder. Pay via {bank} app or net banking you opened.",
            "This is an information SMS from {bank}. There is no action link.",
        ],
    ),
]

LINKS = [
    "http://bit.ly/kyc-upd",
    "https://verify-kyc-help.in/update",
    "https://aadhaar-update.helpdesk-in.com",
    "https://refund-claim.win/in",
    "https://tinyurl.com/kyc24hr",
    "https://bank-secure-login.xyz/kyc",
]
BANKS = ["YourBank", "SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak"]
AMOUNTS = ["99", "199", "487", "1250", "2499", "7800", "18500"]
OTPS = ["184392", "550211", "904176"]
ACCTS = ["4821", "1190", "7733"]
BALANCES = ["2,140.50", "18,332.00", "640.00"]
UPIS = ["kycfine@pay", "refundcell@upi", "officer24@ybl"]


def _fill(template: str, i: int) -> str:
    return template.format(
        link=LINKS[i % len(LINKS)],
        bank=BANKS[i % len(BANKS)],
        amount=AMOUNTS[i % len(AMOUNTS)],
        otp=OTPS[i % len(OTPS)],
        acct=ACCTS[i % len(ACCTS)],
        balance=BALANCES[i % len(BALANCES)],
        upi=UPIS[i % len(UPIS)],
    )


def build_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    bait_extras = [" Reply YES", " Do it in 10 minutes.", ""]
    ok_extras = ["", " If this was not you, open the official app.", " Do not share OTP or PIN."]
    n = 0
    for label, groups in ((1, BAIT), (0, OK)):
        extras = bait_extras if label == 1 else ok_extras
        for category, templates in groups:
            for i, template in enumerate(templates):
                for extra in extras:
                    text = (_fill(template, n) + extra).strip()
                    rows.append(
                        {
                            "text": text,
                            "label": label,
                            "category": category,
                        }
                    )
                    n += 1
    return rows


def main() -> None:
    rows = build_rows()
    SEED_PATH.write_text("", encoding="utf-8")
    with SEED_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "category"])
        writer.writeheader()
        writer.writerows(rows)
    bait = sum(1 for row in rows if row["label"] == 1)
    print(f"Wrote {len(rows)} messages ({bait} bait, {len(rows) - bait} ok) to {SEED_PATH}")


if __name__ == "__main__":
    main()
