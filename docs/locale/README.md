# Translation Guide

This directory holds localization catalogs for Sphinx.

## Workflow
1. Extract messages (English source):
   task docs:gettext
2. Update / create catalogs for target languages (example for es, zh_CN, ja):
   task docs:update-translations
3. Edit the generated `.po` files under `es/LC_MESSAGES`, `zh_CN/LC_MESSAGES`, and `ja/LC_MESSAGES`.
4. Build a localized version (example Spanish):
   sphinx-build -b html -D language=es docs/source docs/_build/html/es

## Conventions
- Keep labels (``.. _label-name:``) untranslated.
- Do not translate code, module paths, or parameter names.
- Prefer concise technical vocabulary consistent with scikit-learn translations.
- If a term has no good direct translation, keep the English word and add a translator comment (`#. NOTE:`) for consistency.

## Useful Flags
- `sphinx-build -b gettext` writes `.pot` sources into `_build/gettext`.
- `sphinx-intl update -p _build/gettext -d docs/locale -l es -l zh_CN -l ja` updates catalogs.

## Quality Checklist Before Commit
- No fuzzy entries left for completed sections.
- No accidental changes to msgid values.
- Build for each language finishes with zero warnings.

## Adding a New Language
1. Add language code to `task docs:update-translations` (or run sphinx-intl manually).
2. Create directory `<lang>/LC_MESSAGES` under `docs/locale`.
3. Re-run extraction + update steps.
