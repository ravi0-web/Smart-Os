import django.template
import numpy as np

# --- 1. Define the register object ONLY ONCE ---
register = django.template.Library()

# --- Utility Functions (using a single register instance) ---

@register.filter
def avg(value_list, key):
    """Calculates the average of a specific key's value in a list of dictionaries."""
    if not value_list:
        return 0.00
    try:
        # Check if the value is a list of Process objects (dicts)
        values = [p.get(key) for p in value_list if p.get(key) is not None]
        if not values:
            return 0.00
        # Use numpy.mean and f-string for formatting
        return f"{np.mean(values):.2f}"
    except Exception:
        return 0.00

@register.filter
def get_item(dictionary, key):
    """
    Retrieves a value from a dictionary using a variable key.
    Usage: {{ my_dictionary|get_item:my_key_variable }}
    """
    if not isinstance(dictionary, dict) or key is None:
        return None
    return dictionary.get(key)

@register.filter
def map(value_list, key):
    """Extracts a list of values for a specific key from a list of dictionaries."""
    if not value_list:
        return []
    # Assumes 'p' is a dictionary.
    return [p.get(key) for p in value_list if isinstance(p, dict) and p.get(key) is not None]

@register.filter
def mean(value_list):
    """Calculates the mean of a list of numbers."""
    if not value_list:
        return 0.0
    try:
        # Use numpy for mean calculation, filtering out None values
        clean_values = [float(v) for v in value_list if v is not None]
        return np.mean(clean_values) if clean_values else 0.0
    except Exception:
        return 0.0

@register.filter
def floatformat(value, precision):
    """Formats a float to a given precision."""
    try:
        # Ensure precision is an integer
        prec = int(precision)
        return f"{float(value):.{prec}f}"
    except (ValueError, TypeError):
        return value

@register.filter
def max(value_list):
    """Returns the maximum value from a list of numbers."""
    if not value_list:
        return 0
    try:
        # Filter out None values and convert to float before finding the max
        clean_values = [float(v) for v in value_list if v is not None]
        return np.max(clean_values) if clean_values else 0
    except Exception:
        return 0

@register.filter
def percentage(value, total):
    """Calculates what percentage 'value' is of 'total', for CSS flex-basis."""
    if total is None or total == 0:
        return 0
    try:
        return (float(value) / float(total)) * 100
    except (ValueError, TypeError):
        return 0

# You don't need the 'get_dict_value' or 'get_key' filters if you use the 
# 'get_item' filter defined above. I'll include the one you defined last 
# for completeness but rename it slightly to avoid conflicts.
@register.filter(name='get_dict_value_alt')
def get_dict_value_alt(dictionary, key):
    """Retrieves a value from a dictionary using a variable key (Alternative to get_item)."""
    try:
        return dictionary[key]
    except (KeyError, TypeError):
        return None
