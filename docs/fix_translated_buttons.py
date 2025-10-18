#!/usr/bin/env python3
"""
Post-processing script to fix translated button text in sphinx-design directives.

This script works around a limitation in sphinx-design's i18n support where
translated button text loses its link/button styling. It finds plain text that
should be buttons and wraps them in the proper HTML structure.

Usage:
    python fix_translated_buttons.py <build_dir> <language>

Example:
    python fix_translated_buttons.py docs/_build/html/zh_CN zh_CN
"""

import sys
from pathlib import Path
from bs4 import BeautifulSoup


# Mapping of English button text to translations and their target URLs
BUTTON_TRANSLATIONS = {
    "zh_CN": {
        # Main index page
        "Discover the environment": {
            "translation": "探索环境",
            "url": "explore/index.html",
        },
        "Chemometrics tutorials": {
            "translation": "化学计量学教程",
            "url": "learn/index.html",
        },
        "Check out our Webinar": {
            "translation": "查看我们的网络研讨会",
            "url": "https://www.youtube.com/watch?v=leB43KchETw&t",
            "external": True,
        },
        # _learn/index page
        "Data sets": {
            "translation": "数据集",
            "url": "datasets.html",
        },
        "Fermentation monitoring": {
            "translation": "发酵监测",
            "url": "pls_regression.html",
        },
        "Coffee classification": {  # Note: typo in original
            "translation": "咖啡分类",
            "url": "pls_classification.html",
        },
        # _explore/index page
        "Get started with scikit-learn": {
            "translation": "开始使用 scikit-learn",
            "url": "sklearn.html",
        },
        "Working with spectra": {
            "translation": "处理光谱数据",
            "url": "spectra.html",
        },
        "Creating your pipelines": {
            "translation": "创建您的流程",
            "url": "pipelines.html",
        },
        "Optimize your processing": {
            "translation": "优化您的处理",
            "url": "optimize.html",
        },
        "DataFrame lover?": {
            "translation": "喜欢 DataFrame？",
            "url": "dataframes.html",
        },
        "Persisting your models": {
            "translation": "持久化您的模型",
            "url": "persist.html",
        },
        # _methods/index page
        "Augmentation": {
            "translation": "数据增强",
            "url": "augmentation.html",
        },
        "Preprocessing": {
            "translation": "预处理",
            "url": "preprocessing.html",
        },
        "Feature selection": {
            "translation": "特征选择",
            "url": "feature_selection.html",
        },
        "Outliers": {
            "translation": "异常值检测",
            "url": "outliers.html",
        },
    },
    "es": {
        # Main index page
        "Discover the environment": {
            "translation": "Descubre el entorno",
            "url": "explore/index.html",
        },
        "Chemometrics tutorials": {
            "translation": "Tutoriales de quimiometría",
            "url": "learn/index.html",
        },
        "Check out our Webinar": {
            "translation": "Mira nuestro seminario web",
            "url": "https://www.youtube.com/watch?v=leB43KchETw&t",
            "external": True,
        },
        # learn/index page
        "Data sets": {
            "translation": "Conjuntos de datos",
            "url": "datasets.html",
        },
        "Fermentation monitoring": {
            "translation": "Monitoreo de fermentación",
            "url": "pls_regression.html",
        },
        "Coffee classification": {  # Note: typo in original
            "translation": "Clasificación de café",
            "url": "pls_classification.html",
        },
        # explore/index page
        "Get started with scikit-learn": {
            "translation": "Comenzar con scikit-learn",
            "url": "sklearn.html",
        },
        "Working with spectra": {
            "translation": "Trabajando con espectros",
            "url": "spectra.html",
        },
        "Creating your pipelines": {
            "translation": "Creando tus pipelines",
            "url": "pipelines.html",
        },
        "Optimize your processing": {
            "translation": "Optimiza tu procesamiento",
            "url": "optimize.html",
        },
        "DataFrame lover?": {
            "translation": "¿Amante de DataFrames?",
            "url": "dataframes.html",
        },
        "Persisting your models": {
            "translation": "Persistiendo tus modelos",
            "url": "persist.html",
        },
        # methods/index page
        "Augmentation": {
            "translation": "Aumento de datos",
            "url": "augmentation.html",
        },
        "Preprocessing": {
            "translation": "Preprocesamiento",
            "url": "preprocessing.html",
        },
        "Feature selection": {
            "translation": "Selección de características",
            "url": "feature_selection.html",
        },
        "Outliers": {
            "translation": "Valores atípicos",
            "url": "outliers.html",
        },
    },
    "ja": {
        # Main index page
        "Discover the environment": {
            "translation": "環境を探索",
            "url": "explore/index.html",
        },
        "Chemometrics tutorials": {
            "translation": "ケモメトリクスのチュートリアル",
            "url": "learn/index.html",
        },
        "Check out our Webinar": {
            "translation": "ウェビナーをチェック",
            "url": "https://www.youtube.com/watch?v=leB43KchETw&t",
            "external": True,
        },
        # learn/index page
        "Data sets": {
            "translation": "データセット",
            "url": "datasets.html",
        },
        "Fermentation monitoring": {
            "translation": "発酵モニタリング",
            "url": "pls_regression.html",
        },
        "Coffee classification": {
            "translation": "コーヒーの分類",
            "url": "pls_classification.html",
        },
        # explore/index page
        "Get started with scikit-learn": {
            "translation": "scikit-learnを始める",
            "url": "sklearn.html",
        },
        "Working with spectra": {
            "translation": "スペクトルデータの操作",
            "url": "spectra.html",
        },
        "Creating your pipelines": {
            "translation": "パイプラインの作成",
            "url": "pipelines.html",
        },
        "Optimize your processing": {
            "translation": "処理の最適化",
            "url": "optimize.html",
        },
        "DataFrame lover?": {
            "translation": "DataFrameが好き？",
            "url": "dataframes.html",
        },
        "Persisting your models": {
            "translation": "モデルの永続化",
            "url": "persist.html",
        },
        # methods/index page
        "Augmentation": {
            "translation": "データ拡張",
            "url": "augmentation.html",
        },
        "Preprocessing": {
            "translation": "前処理",
            "url": "preprocessing.html",
        },
        "Feature selection": {
            "translation": "特徴選択",
            "url": "feature_selection.html",
        },
        "Outliers": {
            "translation": "外れ値検出",
            "url": "outliers.html",
        },
    },
}


def create_button_html(text, url, is_external=False):
    """Create the proper HTML structure for a sphinx-design button."""
    if is_external:
        # External link (button-link)
        classes = "sd-sphinx-override sd-btn sd-text-wrap sd-btn-secondary sd-stretched-link reference external"
        return f'<span class="sd-d-grid"><a class="{classes}" href="{url}"><span>{text}</span></a></span>'
    else:
        # Internal link (button-ref)
        classes = "sd-sphinx-override sd-btn sd-text-wrap sd-btn-secondary sd-stretched-link reference internal"
        return f'<span class="sd-d-grid"><a class="{classes}" href="{url}"><span class="doc">{text}</span></a></span>'


def fix_buttons_in_html(html_path, language):
    """Fix buttons in a single HTML file."""
    if language not in BUTTON_TRANSLATIONS:
        print(f"Warning: No button translations defined for language '{language}'")
        return False

    translations = BUTTON_TRANSLATIONS[language]

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")
    modified = False

    # Find all button links within sd-card-text paragraphs
    for p in soup.find_all("p", class_="sd-card-text"):
        # Look for buttons (links with sd-btn class)
        btn_link = p.find("a", class_="sd-btn")

        if btn_link:
            # Case 1: Button element exists, just translate the text
            span_elem = btn_link.find("span")
            if span_elem:
                english_text = span_elem.get_text(strip=True)

                # Check if we have a translation for this button
                if english_text in translations:
                    translated_text = translations[english_text]["translation"]

                    # Replace the English text with the translation
                    span_elem.string = translated_text

                    modified = True
                    print(
                        f"  ✓ Translated button: '{english_text}' -> '{translated_text}'"
                    )
        else:
            # Case 2: Plain text that should be a button
            # Check if this paragraph contains translated button text
            text_content = p.get_text(strip=True)

            # Look through all translations to find a match
            for english_text, trans_data in translations.items():
                if text_content == trans_data["translation"]:
                    # This is a translated button text without button styling!
                    url = trans_data["url"]
                    is_external = trans_data.get("external", False)

                    # Create the proper button HTML
                    button_html = create_button_html(text_content, url, is_external)

                    # Replace the plain paragraph with the button
                    new_soup = BeautifulSoup(button_html, "html.parser")
                    p.replace_with(new_soup)

                    modified = True
                    print(f"  ✓ Converted plain text to button: '{text_content}'")
                    break

    if modified:
        # Write the modified HTML back
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(str(soup))
        return True

    return False


def process_directory(build_dir, language):
    """Process all HTML files in the build directory."""
    build_path = Path(build_dir)

    if not build_path.exists():
        print(f"Error: Build directory '{build_dir}' does not exist")
        return 1

    html_files = list(build_path.rglob("*.html"))

    if not html_files:
        print(f"Warning: No HTML files found in '{build_dir}'")
        return 0

    print(f"Processing {len(html_files)} HTML files in '{build_dir}'...")
    print(f"Language: {language}\n")

    modified_count = 0
    for html_file in html_files:
        if fix_buttons_in_html(html_file, language):
            print(f"Modified: {html_file.relative_to(build_path)}\n")
            modified_count += 1

    print(f"\nSummary: Modified {modified_count} file(s)")
    return 0


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    build_dir = sys.argv[1]
    language = sys.argv[2]

    return process_directory(build_dir, language)


if __name__ == "__main__":
    sys.exit(main())
