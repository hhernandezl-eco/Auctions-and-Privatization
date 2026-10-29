# Data

`raw/mintic_q4_2020.csv` — operator-level figures transcribed from the MinTIC *Boletín Trimestral de las TIC*, Q4 2020 (published April 2021). Column `source_note` gives the figure (*Gráfico*) number in the bulletin. Units: millions of lines/accesses; COP billion = *miles de millones de pesos* per quarter.

`raw/coltel_2006.csv` — facts on the April 2006 sale of 50 % + 1 share of Colombia Telecomunicaciones, from Infobae (2006-04-08) and Razón Pública.

Derived values (shares, HHI, IMSS, annualised revenue) are computed in `src/privauction/calibration.py`; nothing is hard-coded in the paper.

Known gaps: the bulletin does not report lines for operators outside the four listed (the remaining ~4.6 % of 67.7 million lines is treated as an atomistic fringe); WOM entered in 2021 and is absent from Dec-2020 data.
