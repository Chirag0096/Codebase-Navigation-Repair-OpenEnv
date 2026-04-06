"""Email sending service that uses the template engine."""
from src.template_engine import render_template


def send_welcome_email(user_name: str, user_email: str) -> dict:
    """Send a welcome email to a new user."""
    # BUG: passing 'name' but template_engine expects 'username'
    body = render_template("welcome", name=user_name, email=user_email)
    return {
        "to": user_email,
        "subject": "Welcome!",
        "body": body,
        "sent": True,
    }


def send_reset_email(user_email: str, reset_link: str) -> dict:
    """Send a password reset email."""
    body = render_template("reset", email=user_email, link=reset_link)
    return {
        "to": user_email,
        "subject": "Password Reset",
        "body": body,
        "sent": True,
    }
