# GitHub -> Zenodo -> Imaging Neuroscience release workflow

1. Create a **public GitHub repository** (suggested name: `sgit-radial-angular-meg`) and upload the contents of this folder.
2. Verify the repository locally or in GitHub Actions: `python verify_manifest.py`, `python verify_archive.py`, `python reproduce_article.py`.
3. Connect GitHub to Zenodo, enable the repository in Zenodo, and then create GitHub release/tag **v1.1.0**. Zenodo will archive the release and mint a version DOI.
4. Copy the public GitHub release URL and the **version-specific Zenodo DOI**.
5. Insert both values into the manuscript placeholders `[GITHUB-URL]` and `[ZENODO-DOI]`, and update any desired DOI field in `CITATION.cff`. Recompile the manuscript and combined submission PDF.
6. Submit the combined PDF to *Imaging Neuroscience* via Editorial Manager. Include the same GitHub/Zenodo identifiers in the Data and Code Availability field and cover letter.

Do not publish a Zenodo draft until the uploaded release files and metadata have been checked: Zenodo files cannot be modified after publication except by creating a new version.
