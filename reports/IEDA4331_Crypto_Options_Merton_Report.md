# IEDA4331 Crypto Options Report

The final report is maintained as a LaTeX source and compiled PDF:

- `IEDA4331_Crypto_Options_Merton_Report.tex`
- `build/IEDA4331_Crypto_Options_Merton_Report.pdf`

The older HTML draft was superseded because browser-side equation rendering was unreliable for the LaTeX-heavy methodology sections.

Rebuild the PDF from the `reports/` directory:

```bash
/Users/marco/.codex/plugins/cache/openai-bundled/latex-tectonic/0.1.1/bin/tectonic --outdir build IEDA4331_Crypto_Options_Merton_Report.tex
```

Regenerate figures and statistical tables from the project root:

```bash
.venv/bin/python scripts/generate_report_assets.py
```

