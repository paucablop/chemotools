What is chemotools?
====================

``chemotools`` is an open-source Python package for spectral preprocessing and chemometric modelling. It integrates directly with the ``scikit-learn`` ecosystem, enabling reproducible and scalable development of spectroscopic models.

* **Extensive toolkit:** transformers for diverse spectroscopic data.  
* **Composable and scalable:** pipelines that integrate directly and grow with your workflow.  
* **Reliable and transparent:** rigorously tested, peer-reviewed, and open.  


.. note::
   Build exactly what you need, integrate with the tools you trust, and scale without compromise.

.. image:: ./_static/devops-vectorized.png
   :alt: Composable building blocks / lifecycle
   :align: center
   :width: 300px

.. hint::
   **Curious about scikit-learn?** See the dedicated `scikit-learn overview <../explore/sklearn.html>`_.

Extensive Toolkit
-----------------
Spectroscopy is diverse — different **instruments**, **sample matrices**, and **analysis goals** demand different approaches. ``chemotools`` provides a **focused set of transformers** for spectroscopic data and makes it simple to assemble them into **preprocessing pipelines**. By connecting to the wider **Python ecosystem**, you can leverage **state-of-the-art machine learning** and **scientific libraries** to build **robust, end-to-end spectroscopic models**.

.. image:: ./_static/atom.png
   :class: no-background
   :alt: Ecosystem: chemotools with sklearn, numpy and friends
   :align: center
   :width: 340px

.. note::
   Connect your preprocessing with the ``scikit-learn`` ecosystem to unlock a wide toolstack for feature selection, model tuning, validation, and deployment.

.. hint::
   **Want to explore the tools at your fingertips?** Visit the `explore overview <../explore/index.html>`_.

Composable and Scalable
-----------------------
``chemotools`` transforms preprocessing into a **modular, composable process**: each transformer follows the ``fit`` / ``transform`` API and integrates directly with ``scikit-learn`` pipelines. Preprocessing, feature selection, modelling, and validation can be chained directly — no adapters required.  

This composability ensures **reproducibility** (pipelines are declarative), **robustness** (swap blocks without refactoring), and **scalability** (parallelization and deployment patterns leveraging already supported infrastructure by the ecosystem).

.. note::
   From exploration to deployment, ``chemotools`` grows with your workflow.

Quality & Transparency
-----------------------
``chemotools`` combines extensibility and scalability with open-source transparency, making validation and trust easier from the start.

* **Thorough testing** — covered by extensive `unit tests <https://app.codecov.io/github/paucablop/chemotools>`_ and continuous code coverage for reliability and robustness.  
* **Transparency** — we provide a Software Bill of Materials (SBOM) in `CycloneDX <https://cyclonedx.org/>`_ format with every release, supporting validation and clearance in regulated environments.  
* **Peer-reviewed** — published in the `Journal of Open Source Software (JOSS) <https://joss.theoj.org/papers/10.21105/joss.06802>`_ following independent peer review.  

.. note::
   Quality, transparency, and trust — built in from the start.
