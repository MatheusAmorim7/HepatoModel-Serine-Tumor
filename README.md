# HepatoModel: hepatic serine supply and tumour demand

Reproduce the separate metabolic tests behind Figures 2 and 3 of the SysBioRio poster. This is not the spatial PhysiCell simulation.

## What we did

Hepatocyte metabolic networks and a tumour module share a finite, well-mixed medium for 12 hours. We compare allowing or blocking hepatic serine export, changing external nutrients and the tumour HPRT reaction. The optimiser maximises prescribed task fulfilment or minimises the additional hepatic serine required for a fixed task target.

The tests ask when purine salvage can spare serine, and when material serine use or extra one-carbon demand still requires support. The parameters are exploratory. Task fulfilment does not demonstrate tumour growth or drug efficacy.

## Run

For Google Colab, open `colab.ipynb` and run the cells in order. It downloads the repository without a manual upload.

Python 3.12 is the tested interpreter. Use a new virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. The GLPK Python package is included in the requirements. No Gurobi licence is needed.

The default run recomputes the poster examples, boundary checks and selected nutrient controls. It checks the results against archived numerical references, reruns raw objectives in fresh models and checks mass balance, finite stocks, carbon/nitrogen and ATP accounting. It then produces Figures 2 and 3 from the new calculations.

```bash
python reproduce.py --full --out-dir results_full
```

The full option repeats all 2,200 scenarios from the demand, supply and nutrient-context stages. Previous output directories are never overwritten: choose a new `--out-dir` when rerunning.

## Files

- `results/results.csv`: newly computed cases and flux summaries.
- `results/stocks.csv`: nutrient inventories for each feasible case.
- `results/objective_checks.csv` and `results/checks.json`: numerical checks.
- `results/figure2_demand.*` and `results/figure3_supply_HPRT.*`: figures.
- `reference_results/`: archived values used for regression testing, not plot inputs.
- `notebooks/`: original implementation modules, copied unchanged. Run `reproduce.py`, not their historical standalone entry points.
- `source_manifest.json`: input/source hashes and the tested dependency versions.

## Biological scope

The tumour module contains selected HepatoNet1 chemistry for serine, folates, purines and energy. Hepatocytes carry metabolic networks and pay maintenance costs. The model uses 252 basal hepatocyte equivalents, 12 potential suppliers and 7 tumour equivalents in 20 nL, not a growing spatial tumour.

At 100% task, the aggregate material demands are serine 0.056, IMP 0.014 and additional C1 0.014 pmol/h. ATP work is 2.3912 times the task fraction, plus fixed maintenance of 0.0035 pmol/h and reaction-specific costs. The 55% target is an illustrative fraction between two reference capacities, not a measured growth rate.

HPRT blocked means zero flux through that reaction. It does not represent all purine salvage being blocked. HPRT activity and hepatic serine export are separate controls. The poster protocol closes tumour serine synthesis and local formate export to isolate salvage. The results are conditional on those choices.

No questioned Nature source-data panels or private author correspondence are packaged as independent validation. The Parker serine exchange estimate is an exploratory transport reference from another culture context, not a measured tumour Vmax.

## References

- Gille et al. HepatoNet1 (2010). https://doi.org/10.1038/msb.2010.62
- Ebrahim et al. COBRApy (2013). https://doi.org/10.1186/1752-0509-7-74
- Parker et al. (2020). https://doi.org/10.1158/2159-8290.CD-19-0959
- Hennequart et al. (2021). https://doi.org/10.1038/s41467-021-26395-5
- Ganguly et al. (2026). https://doi.org/10.1038/s41586-026-11051-z
- Dolfi et al. (2013), generic ATP-cost reference. https://doi.org/10.1186/2049-3002-1-20

Third-party models and supplementary data retain their original attribution and applicable terms. This repository does not grant a new licence over those materials. The local `hepatonet_original_13_08.xml` is the input used by this project, not a claim of an unmodified official distribution.

## Resumo em portugues

Este pacote reproduz os testes metabólicos separados do lóbulo. Hepatócitos e tumor compartilham nutrientes finitos. Comparamos a necessidade de serina hepática com diferentes demandas, oferta externa e capacidade de HPRT. Rode `python reproduce.py` para os exemplos do pôster ou adicione `--full` para todos os cenários. Os resultados são exploratórios e não representam crescimento tumoral validado.
