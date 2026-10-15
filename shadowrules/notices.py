"""Step 6 - adverse-action notice consistency.

Reg B requires the reasons stated on an adverse-action notice to be the
principal reasons the application was actually declined. Using the formalized
policy clauses, each declined file's stated reason is classified:

  consistent   the reason maps to a policy clause the file actually fails
  inaccurate   the reason maps to a policy clause the file *passes*
               (told "DTI too high" with a DTI the written policy accepts)
  off-policy   the reason names a criterion the written policy does not contain
  missing      no reason was recorded at all
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import logic


def classify(book: pd.DataFrame, forms) -> np.ndarray:
    by_reason = {f.clause.reason_code: logic.node_mask(f.node, book) for f in forms}
    reason = book["stated_reason"].fillna("").to_numpy()
    declined = book["approve"].to_numpy() == 0
    status = np.full(len(book), "", dtype=object)
    for i in np.flatnonzero(declined):
        passes = by_reason.get(reason[i])
        if not reason[i]:
            status[i] = "missing"
        elif passes is None:
            status[i] = "off-policy"
        else:
            status[i] = "inaccurate" if passes[i] else "consistent"
    return status
