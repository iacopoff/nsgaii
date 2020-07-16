import numpy as np
import pandas as pd



x = (("A",1,2,3),
     ("A",1,2,3),
     ("A",1,2,3),
     ("B",1,2,3),
     ("A",1,1,1),
     ("C",1,2,3),
     ("B",1,2,3)
)

df = pd.DataFrame(x)

df_nodup = df[~df.duplicated()]
df_dup = df[df.duplicated()]


import pdb; pdb.set_trace()
