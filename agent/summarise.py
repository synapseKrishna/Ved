import os
import ssl
import smtplib
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr


import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

# Email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "synapsebiopharma@gmail.com"
SENDER_NAME = "Vedolizumab Bot"
RECIPIENTS = ["divyanshmi999@gmail.com","krishna@synapsebiopharma.com",
"utkarsh@synapsebiopharma.com",
"diksha@synapsebiopharma.com",
"dipanshi@synapsebiopharma.com",
"navita@synapsebiopharma.com",
"satish@synapsebiopharma.com"]


# Explicit (testing)
EMAIL_PASSWORD ="nwtungnsbcndsxne"

# ==========================================================
# EMAIL (HTML + TEXT)
# ==========================================================

def send_consolidated_email(df: pd.DataFrame) -> bool:
    if df.empty:
        print("No relevant news to send.")
        return False

    msg = EmailMessage()
    msg["From"] = formataddr((SENDER_NAME, SENDER_EMAIL))
    msg["To"] = ", ".join(RECIPIENTS)
    
    timestamp = datetime.now().strftime("%d %b %Y")
    msg["Subject"] = f"Vedolizumab Update - {timestamp}"

    # Group by company
    grouped = df.groupby("company")

    # Plain text fallback
    text_lines = ["FYIP\n", f"Vedolizumab Update - {timestamp}\n"]
    
    for company, group in grouped:
        text_lines.append(f"--- {company.upper()} ---")
        for _, row in group.iterrows():
            headline = row.get("title", "No Title")
            source_url = row.get("link", "#")
            source = row.get("source", "")
            text_lines.append(f"• {headline}")
            text_lines.append(f"  Source: {source} | Link: {source_url}\n")
        text_lines.append("")
        
    text_lines.append("Regards,\nSynapse News Bot")
    text_body = "\n".join(text_lines)
    msg.set_content(text_body)

    # HTML version (Outlook Safe)
    html_parts = [
        '<html>',
        '<body style="font-family: Arial, sans-serif; font-size:14px; color: #333333; line-height: 1.6;">',
        '<table width="100%" cellpadding="0" cellspacing="0" border="0">',
        '<tr><td align="center">',
        '<table width="100%" style="max-width: 800px; padding: 20px;" cellpadding="0" cellspacing="0" border="0">',
        '<tr><td align="left" style="font-family: Arial, sans-serif;">',
        '<p style="font-size: 14px; color: #555;">FYIP,</p>',
        f'<h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-family: Arial, sans-serif;">Vedolizumab Update - {timestamp}</h2>'
    ]

    for company, group in grouped:
        html_parts.append(f'<h3 style="color: #e67e22; margin-top: 30px; margin-bottom: 15px; font-size: 18px; font-family: Arial, sans-serif;">{company}</h3>')
        
        for _, row in group.iterrows():
            headline = row.get("title", "No Title")
            source_url = row.get("link", "#")
            source = row.get("source", "Unknown Source")
            tag = row.get("tag", "News")
            
            html_parts.append(
                f'<table width="100%" cellpadding="15" cellspacing="0" border="0" style="margin-bottom: 15px; background-color: #f8f9fa; border-left: 4px solid #3498db;">'
                f'<tr><td style="font-family: Arial, sans-serif;">'
                f'<p style="margin: 0 0 8px 0; font-size: 16px; font-weight: bold; color: #2c3e50;">{headline}</p>'
                f'<p style="margin: 0; font-size: 13px;">'
                f'<span style="background-color: #e1f0fa; color: #2980b9; padding: 2px 6px; font-weight: bold;">{tag}</span>'
                f'&nbsp;&nbsp;'
                f'<span style="color: #7f8c8d;">{source}</span>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<a href="{source_url}" target="_blank" style="color: #2980b9; text-decoration: none; font-weight: bold;">Read Article &rarr;</a>'
                f'</p>'
                f'</td></tr></table>'
            )

    html_parts.append('<br><p style="margin-top: 30px; color: #555; font-family: Arial, sans-serif;">Regards,<br><strong style="color: #2c3e50;">Synapse News Bot</strong></p>')
    html_parts.append('</td></tr></table>')
    html_parts.append('</td></tr></table>')
    html_parts.append('</body></html>')
    
    html_body = "\n".join(html_parts)
    msg.add_alternative(html_body, subtype="html")

    # ==========================================================
    # ATTACH EXCEL FILES
    # ==========================================================

    attachments = [
        "news_results_raw.xlsx",
        "news_results.xlsx"
    ]

    for file in attachments:
        if os.path.exists(file):
            with open(file, "rb") as f:
                file_data = f.read()

            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=os.path.basename(file)
            )

            print(f"📎 Attached: {file}")
        else:
            print(f"⚠️ Attachment not found: {file}")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

# ==========================================================
# MAIN PROCESS
# ==========================================================

def process_articles(input_file="news_results.xlsx"):
    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ Failed to read {input_file}: {e}")
        return

    df.columns = df.columns.str.lower().str.strip()

    # 🚫 Exclude Financial and Other tags
    if "tag" in df.columns:
        df = df[~df["tag"].isin(["Financial", "Other"])]

    # Drop rows without a valid link
    df = df.dropna(subset=["link"])
    df = df[df["link"].astype(str).str.startswith("http")]

    print(f"📊 Compiling {len(df)} relevant articles into a single email and WhatsApp message...")

    if send_consolidated_email(df):
        print("✅ Consolidated email sent successfully!")
    else:
        print("⚠️ No email was sent.")
        
    # from agent.whatsapp_alert import send_whatsapp_alert
    # if send_whatsapp_alert(df):
    #     print("✅ WhatsApp alert sent successfully!")
    # else:
    #     print("⚠️ WhatsApp alert was not sent.")

if __name__ == "__main__":
    process_articles()