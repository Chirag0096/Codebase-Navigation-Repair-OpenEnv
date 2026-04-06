"""Template rendering engine for email bodies."""

TEMPLATES = {
    "welcome": "Hello {username}, welcome to our platform! Your email {email} has been registered.",
    "reset": "Click here to reset your password: {link}. This was requested for {email}.",
    "notify": "Hi {username}, you have a new notification: {message}.",
}


def render_template(template_name: str, **kwargs) -> str:
    """
    Render an email template with the given keyword arguments.
    
    Expected kwargs per template:
    - welcome: username (str), email (str)
    - reset: email (str), link (str)
    - notify: username (str), message (str)
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = TEMPLATES[template_name]
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing required template variable: {e}")
