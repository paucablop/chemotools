{{ objname }}
{{ "=" * objname|length }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
