from twilio.rest import Client

def send_emergency_sms(message_body):
    # Credentials from Twilio Console
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message_body,
        from_='', # Your Twilio number
        to=''    # Your emergency contact
    )
    print(f"📩 SMS Sent: {message.sid}")