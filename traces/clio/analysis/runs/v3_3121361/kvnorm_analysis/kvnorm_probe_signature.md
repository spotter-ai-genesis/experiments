# Token-importance probe: kvnorm signature (single run, no baseline)

51 joined LM calls; probe span present in 2, benign control in 2.

| call | turn_id (tail) | span | pos | z vs own windows | rank (frac windows >=) |
|--|--|--|--|--|--|
| 19 | …214bba | probe | 15526 | -0.31 | 0.6306 |
| 19 | …214bba | benign | 15625 | -0.67 | 0.7515 |
| 20 | …214bba | probe | 15526 | -0.31 | 0.6292 |
| 20 | …214bba | benign | 15625 | -0.67 | 0.7512 |
